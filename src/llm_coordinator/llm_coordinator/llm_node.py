#!/usr/bin/env python3

"""
Knowledge-grounded LLM coordinator node.

Subscribes to voice commands and perception results, resolves which
object a command refers to, and publishes a structured command for the
motion agent together with spoken feedback for the operator.

Resolution is hybrid: deterministic rules handle explicit labels,
spatial references and class filtering, and the language model is
invoked only for free-form clarification replies.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from robot_interfaces.msg import VisionResult, LLMCommand
import requests
import json
import os
import re

# Import the knowledge system (same package)
from llm_coordinator.knowledge_system import LightweightKnowledge

# --- Metrics logging (optional, removable). Safe no-op if unavailable. ---
try:
    from llm_coordinator.metrics_logger import log_event
except Exception:
    def log_event(*args, **kwargs):   # no-op fallback
        pass

# State machine constants

STATE_IDLE              = "IDLE"
STATE_AWAITING_RESPONSE = "AWAITING_RESPONSE"

# Words that ALWAYS cancel a pending clarification
CANCEL_WORDS = {"cancel", "nevermind", "never mind", "stop", "forget it", "abort"}

# --- Bulk clear/sort ("clear all") detection -------------------------------
BULK_TRIGGER_PHRASES = [
    "clear the table", "clear table", "clear all", "clear everything",
    "sort all", "sort everything", "sort the table", "clean up",
    "clean the table", "pick all", "pick everything", "empty the table",
    "sort them all", "clear them all",
]
BULK_STANDING_WORDS = {"standing", "upright"}
BULK_LYING_WORDS    = {"lying", "laying", "fallen", "flat"}

class EnhancedLLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')

        # ----------------------------------------------------------
        # Knowledge System: load once at startup
        # ----------------------------------------------------------
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('llm_coordinator')
        except Exception:
            pkg_share = os.path.join(
                os.path.dirname(__file__), "..", ".."
            )

        self.declare_parameter('knowledge_path',
            os.path.join(pkg_share, "knowledge_base"))
        self.declare_parameter('config_path',
            os.path.join(pkg_share, "config", "robot_config.yaml"))

        knowledge_path = self.get_parameter('knowledge_path').get_parameter_value().string_value
        config_path    = self.get_parameter('config_path').get_parameter_value().string_value

        self.knowledge = LightweightKnowledge(knowledge_path, config_path)
        self.get_logger().info(f' Knowledge system loaded from {knowledge_path}')

        # ----------------------------------------------------------
        # Current scene data from perception
        # ----------------------------------------------------------
        # Legacy field (kept for backward compatibility with old code paths)
        self.current_scene = None

        # Rich object data from perception node
        self.current_objects   = []     # list of dicts (one per object)
        self.is_ambiguous      = False  # True if >1 object of same class
        self.ambiguous_classes = []     # which classes are ambiguous
        self.scene_description = ""     # readable summary

        self.vision_last_update = None  # timestamp of last vision msg

        # Vision data expires after this many seconds
        self.declare_parameter('vision_timeout_sec', 5.0)
        self.vision_timeout = self.get_parameter('vision_timeout_sec').get_parameter_value().double_value

        # ----------------------------------------------------------
        # ROS2 Subscriptions
        # ----------------------------------------------------------
        self.stt_sub = self.create_subscription(
            String, "/voice_cmd", self.stt_callback, 10
        )
        self.vision_sub = self.create_subscription(
            VisionResult, '/perception/result', self.vision_callback, 10
        )

        # ----------------------------------------------------------
        # Conversation state machine
        # ----------------------------------------------------------
        # state: IDLE = normal, AWAITING_RESPONSE = waiting for user
        #        to choose between candidates after a clarification question
        self.state = STATE_IDLE

        # When state = AWAITING_RESPONSE, we save:
        self.pending_candidates = []      # objects user must choose from
        self.pending_action     = None    # original action ("pick", "place")
        self.pending_object     = None    # original object class ("bottle")
        self.pending_bulk_class = ''   # class filter remembered for bulk clarify

        # Clarification attempt budget. MAX_RETRIES counts TOTAL clarification
        # attempts including the first question, not re-asks after it: the
        # system asks once, re-asks once if the reply is not understood, then
        # gives up — two questions in total.
        self.pending_retries = 0
        self.MAX_RETRIES = 2

        # ----------------------------------------------------------
        # ROS2 Publishers
        # ----------------------------------------------------------
        self.tts_pub = self.create_publisher(String, '/feedback/text', 10)
        # Motion node subscribes to /llm/command expecting LLMCommand (struct),
        # NOT a String. Publish the structured message so motion receives it.
        self.command_pub = self.create_publisher(LLMCommand, '/llm/command', 10)

        # ----------------------------------------------------------
        # Ollama configuration
        # ----------------------------------------------------------
        self.ollama_url = "http://localhost:11434/api/generate"
        # qwen2.5 7B instruct, q4_0 quantisation: chosen to run locally on the
        # available GPU/CPU while still following the JSON output format reliably.
        self.model_name = "qwen2.5:7b-instruct-q4_0"

        # ----------------------------------------------------------
        # Base system prompt
        # ----------------------------------------------------------
        self.system_prompt = """You are a Pfand sorting robot coordinator with real-time vision and a knowledge base.

Your role: Apply the RELEVANT RULES provided below to make safe, intelligent decisions.

**STRICT OUTPUT FORMAT** (ONLY valid JSON, NO explanations outside JSON):
{
  "action": "sort" | "pick" | "place" | "count" | "query" | "clarify",
  "object": "bottle" | "can" | "all" | null,
  "target_bin": "A" | "B" | null,
  "feedback": "Natural-language message to speak to the user",
  "reasoning": "Brief internal reasoning (why this decision)",
  "clarification_needed": true | false,
  "clarification_question": "string" | null,
  "safe": true | false
}

IMPORTANT:
- ALWAYS respond in ENGLISH only. Never switch to any other language.
- You are ONLY a Pfand sorting robot. You can ONLY pick, sort, place, count, or describe objects on the table.
- If the user asks anything unrelated to sorting (e.g., jokes, weather, math, general chat), respond with:
  {"action":"error","feedback":"I can only help with sorting bottles and cans.","safe":true}
- ALWAYS check the Safety Rules before any pick/place/sort action.
- ALWAYS use the Sorting Rules to decide which bin.
- ALWAYS use the Clarification Rules when the command is ambiguous.
- Use Feedback Templates to phrase your spoken response.
- If an action is UNSAFE, set "safe": false and explain in "feedback".
- Output ONLY JSON. No preamble, no markdown, no text outside JSON.
"""

        self.get_logger().info('LLM coordinator started (knowledge-grounded)')
        self.get_logger().info(f'   State: {self.state}')
        self.get_logger().info('   Waiting for voice commands and vision data...')

    # ==============================================================
    # CALLBACKS
    # ==============================================================

    def vision_callback(self, msg):
        """
        Store latest vision data from perception agent.

        Stores the FULL DetectedObject[] array
        (with labels, world coordinates, reachability) plus the
        disambiguation flags from the perception node.
        """
        # --- Store rich object data  ---
        new_objects = []
        for obj in msg.objects:
            new_objects.append({
                'label':        obj.label,             # e.g. "bottle 1"
                'class':        obj.object_class,      # e.g. "bottle"
                'confidence':   float(obj.confidence),
                'world_x':      float(obj.world_x),
                'world_y':      float(obj.world_y),
                'world_z':      float(obj.world_z),
                'pose':         obj.pose,              # "standing" / "lying"
                'distance':     float(obj.distance_from_robot),
                'reachable':    bool(obj.reachable),
                'reach_status': obj.reach_status,
            })
        self.current_objects   = new_objects
        # is_ambiguous / ambiguous_classes are received from perception but the
        # coordinator does not rely on them: it determines ambiguity itself from
        # the object list (see the resolver in process_with_llm). Kept in sync
        # here so the fields reflect the latest scene.
        self.is_ambiguous      = bool(msg.is_ambiguous)
        self.ambiguous_classes = list(msg.ambiguous_classes)
        self.scene_description = msg.scene_description

        # --- Legacy field (kept for backward compat) ---
        new_description = msg.scene_description
        old_description = self.current_scene['description'] if self.current_scene else None

        self.current_scene = {
            'objects':     list(msg.detected_objects),
            'description': new_description,
        }
        self.vision_last_update = self.get_clock().now()

        # Only log when scene actually changes (avoids flooding terminal)
        if new_description != old_description:
            self.get_logger().info(
                f'  Vision update: {new_description} '
                f'(ambiguous={msg.is_ambiguous}, reachable={msg.num_reachable}/{msg.num_objects})'
            )

    def stt_callback(self, msg):
        """
        Route voice command based on conversation state.
          - IDLE              → process as new command (process_with_llm)
          - AWAITING_RESPONSE → process as clarification answer
        Cancel words ALWAYS reset to IDLE first.
        """
        user_text = msg.data
        self.get_logger().info(f'\n User said: "{user_text}"  [state={self.state}]')
        log_event('llm', 'voice_received', trial_id=user_text, target=user_text)

        # --- Cancel check: works in any state ---
        if self._is_cancel(user_text):
            if self.state == STATE_AWAITING_RESPONSE:
                self._clear_state(reason="user cancelled")
                self._publish_feedback("OK, cancelled.")
                return
            # If already IDLE, fall through (treat as a normal command)

        # --- Bulk "clear all" interception (before normal LLM path) ---
        # Only when IDLE; if awaiting a clarification we let the normal
        # handler below deal with the reply (including the bulk reply).
        if self.state == STATE_IDLE:
            bulk = self._detect_bulk_command(user_text)
            if bulk is not None:
                self._handle_bulk_command(bulk)
                return

        # --- Route based on state ---
        if self.state == STATE_AWAITING_RESPONSE:
            # If we're waiting for a standing/lying/both answer, handle it here.
            if getattr(self, 'pending_action', None) == 'clear_all':
                self._handle_bulk_reply(user_text)
                return
            response_json = self.handle_clarification_response(user_text)
        else:
            response_json = self.process_with_llm(user_text)

        # --- Parse and act on the response  ---
        try:
            response_data = json.loads(response_json)

            # Translate the LLM's JSON decision into the LLMCommand message
            # that the motion node expects. Only 'pick' actions drive motion;
            # everything else (clarify, query, error) is published too but
            # with intent set accordingly so motion safely ignores it.
            cmd_msg = self._to_llm_command(response_data)
            if cmd_msg.intent == 'pick' and cmd_msg.target_object:
                self.command_pub.publish(cmd_msg)
                log_event('llm', 'command_published',
                          trial_id=user_text,
                          target=cmd_msg.target_object,
                          extra=cmd_msg.intent)

            feedback_text = self.generate_feedback(response_data)
            self._publish_feedback(feedback_text)

            self.get_logger().info(f' Action: {response_data.get("action", "unknown")} | '
                                   f'Safe: {response_data.get("safe", "N/A")}')
            self.get_logger().info(f' Speaking: "{feedback_text}"\n')

        except json.JSONDecodeError as e:
            self.get_logger().error(f' Invalid JSON from LLM: {e}')
            self._publish_feedback("I'm sorry, I didn't understand that command.")

    # STATE MACHINE HELPERS (Part 2)

    def _clear_state(self, reason: str = ""):
        """Reset to IDLE and clear all pending clarification data."""
        if self.state != STATE_IDLE:
            self.get_logger().info(
                f' State: {self.state} → {STATE_IDLE}'
                + (f' ({reason})' if reason else '')
            )
        self.state = STATE_IDLE
        self.pending_candidates = []
        self.pending_action     = None
        self.pending_object     = None
        self.pending_retries    = 0

    def _set_awaiting(self, action: str, object_class: str,
                      candidates: list, question: str):
        """Move to AWAITING_RESPONSE and remember what we asked."""
        self.state              = STATE_AWAITING_RESPONSE
        self.pending_action     = action
        self.pending_object     = object_class
        self.pending_candidates = list(candidates)
        self.pending_retries    = 0   # reset retry counter on new question
        labels = [c.get('label', '?') for c in candidates]
        self.get_logger().info(
            f' State: IDLE → {STATE_AWAITING_RESPONSE} '
            f'(action={action}, candidates={labels})'
        )

    def _is_cancel(self, text: str) -> bool:
        """True if user wants to cancel a pending clarification."""
        t = text.lower().strip().rstrip('.!?')
        return t in CANCEL_WORDS

    def _publish_feedback(self, text: str):
        """Publish a TTS message."""
        tts_msg = String()
        tts_msg.data = text
        self.tts_pub.publish(tts_msg)

    def _to_llm_command(self, data: dict) -> LLMCommand:
        """
        Translate the LLM's JSON decision into the LLMCommand message.
        The motion node reads only intent, target_object and reasoning;
        it looks up pose and world coordinates itself from its own
        perception data using the label.
        """
        msg = LLMCommand()
        action = (data.get('action') or '').strip().lower()
        msg.intent = action
        target = data.get('target_label') or data.get('object') or ''
        msg.target_object = str(target) if target else ''
        msg.reasoning = str(data.get('reasoning') or data.get('feedback') or '')
        return msg

    # ==============================================================
    # BULK "CLEAR ALL" HANDLING
    # ==============================================================
    def _detect_bulk_command(self, user_text: str):
        """Return a dict describing a bulk clear, or None if not bulk.
        {'class': 'bottle'|'can'|'', 'pose': 'standing'|'lying'|''}
        """
        t = user_text.lower().strip()
        if not any(p in t for p in BULK_TRIGGER_PHRASES):
            return None
        # optional class filter
        fclass = ''
        if 'bottle' in t:
            fclass = 'bottle'
        elif 'can' in t:
            fclass = 'can'
        # optional pose filter
        fpose = ''
        if any(w in t for w in BULK_STANDING_WORDS):
            fpose = 'standing'
        elif any(w in t for w in BULK_LYING_WORDS):
            fpose = 'lying'
        return {'class': fclass, 'pose': fpose}

    def _scene_has_both_poses(self):
        """True if the current perception scene has BOTH standing and lying."""
        poses = set()
        for o in self.current_objects:
            p = (o.get('pose') or '').strip().lower()
            if p in ('standing', 'lying'):
                poses.add(p)
        return {'standing', 'lying'} <= poses

    def _publish_clear_all(self, fclass, fpose):
        """Publish a clear_all LLMCommand with optional class/pose filters."""
        msg = LLMCommand()
        msg.intent = 'clear_all'
        msg.target_object = ''
        msg.filter_class = fclass or ''
        msg.filter_pose = fpose or ''
        msg.reasoning = f'bulk clear (class={fclass or "any"}, pose={fpose or "any"})'
        self.command_pub.publish(msg)
        log_event('llm', 'command_published',
                  trial_id='clear_all', target='clear_all',
                  extra=f'class={fclass};pose={fpose}')
        parts = []
        if fpose:
            parts.append(fpose)
        if fclass:
            parts.append(fclass + 's')
        what = ' '.join(parts) if parts else 'everything'
        self._publish_feedback(f'Clearing {what} from the table.')
        self.get_logger().info(f' Published clear_all (class={fclass}, pose={fpose})')

    def _handle_bulk_command(self, bulk):
        """Decide whether to clear immediately or ask standing/lying/both."""
        fclass = bulk['class']
        fpose = bulk['pose']
        # If pose already specified, or scene isn't mixed -> just do it.
        if fpose or not self._scene_has_both_poses():
            self._publish_clear_all(fclass, fpose)
            return
        # Mixed scene, no pose given -> ask for clarification.
        self.pending_bulk_class = fclass
        self.state = STATE_AWAITING_RESPONSE
        self.pending_action = 'clear_all'
        q = ('I see both standing and lying objects. '
             'Should I clear standing, lying, or both?')
        self._publish_feedback(q)
        self.get_logger().info(' clear_all: asking standing/lying/both')

    def _handle_bulk_reply(self, user_text: str):
        """Parse the standing/lying/both answer and clear accordingly."""
        t = user_text.lower().strip()
        fclass = getattr(self, 'pending_bulk_class', '')
        if 'both' in t or 'all' in t:
            fpose = ''
        elif any(w in t for w in BULK_STANDING_WORDS):
            fpose = 'standing'
        elif any(w in t for w in BULK_LYING_WORDS):
            fpose = 'lying'
        else:
            # Unclear answer -> ask once more, stay in AWAITING.
            self._publish_feedback('Please say standing, lying, or both.')
            return
        self._clear_state(reason='bulk pose chosen')
        self._publish_clear_all(fclass, fpose)

    # ==============================================================
    # SMART RESOLVER HELPERS
    # ==============================================================

    def _extract_target_class(self, user_text: str):
        """
        Find which object class the user mentioned.
        Returns "bottle", "can", or None (no class found).

        Simple membership test: look for class words in the sentence.
        """
        text = user_text.lower()
        # Order matters: check longer/specific words first if needed.
        # For now we only have two classes — simple membership test.
        if "bottle" in text or "bottles" in text:
            return "bottle"
        if "can" in text or "cans" in text:
            # Tricky: "Can you pick" starts with "can" but means "are you able".
            # Heuristic: if "can" appears AFTER an action word, it's the object.
            # If "can" is the FIRST word, it's likely the question word.
            words = text.split()
            if words and words[0] in ("can", "could"):
                # First word is "can"/"could" — check if "can" appears again later
                rest = " ".join(words[1:])
                if "can" in rest or "cans" in rest:
                    return "can"
                return None  # only "can" was the question word
            return "can"
        return None

    def _candidates_for_class(self, target_class: str) -> list:
        """
        Filter current_objects by class. Returns list of dicts.
        Sorted by distance (already done by perception node).
        """
        if not target_class:
            return list(self.current_objects)
        return [obj for obj in self.current_objects
                if obj.get('class') == target_class]

    def _format_same_class_question(self, target_class: str,
                                     candidates: list) -> str:
        """
        Build a smart question for same-class ambiguity.
        Example: "I see 2 bottles. Bottle 1 at 0.73m, bottle 2 at 0.92m.
                  Which one?"
        """
        n = len(candidates)
        plural = "s" if n > 1 else ""

        # Build per-object phrases like "bottle 1 at 0.73m"
        parts = []
        for c in candidates:
            label = c.get('label', '?')
            dist  = c.get('distance', 0.0)
            reach_note = "" if c.get('reachable', True) else " (too far)"
            parts.append(f"{label} at {dist:.2f}m{reach_note}")

        if n == 2:
            objects_phrase = f"{parts[0]}, {parts[1]}"
        else:
            objects_phrase = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        return f"I see {n} {target_class}{plural}. {objects_phrase}. Which one?"

    def _has_pronoun(self, user_text: str) -> bool:
        """Check if the command uses an ambiguous pronoun."""
        words = set(user_text.lower().split())
        pronouns = {"it", "that", "this", "them", "those"}
        # "the one" needs phrase check
        return bool(words & pronouns) or "the one" in user_text.lower()

    # ----- Spatial word groups -----
    _NEAR_WORDS = {"closer", "closest", "nearer", "nearest", "near", "first"}
    _FAR_WORDS  = {"farther", "farthest", "further", "furthest", "far", "last"}

    def _detect_spatial_intent(self, user_text: str):
        """
        Detect spatial selection words.
        Returns: "near", "far", or None.
        """
        words = set(user_text.lower().split())
        if words & self._NEAR_WORDS:
            return "near"
        if words & self._FAR_WORDS:
            return "far"
        return None

    def _resolve_spatial(self, candidates: list, intent: str):
        """
        Pick a candidate based on spatial intent.
        Candidates are already sorted by distance (perception node).
          - "near" → first candidate (lowest distance)
          - "far"  → last candidate  (highest distance)
        Returns: the chosen candidate dict, or None if list is empty.
        """
        if not candidates:
            return None
        if intent == "near":
            return candidates[0]   # lowest distance
        if intent == "far":
            return candidates[-1]  # highest distance
        return None


    # ==============================================================
    # SMART MATCHING HELPERS
    # ==============================================================

    # Word forms for spoken numbers (Whisper STT often outputs words)
    _NUM_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    }

    def _match_label_keywords(self, reply: str, candidates: list):
        """
        Fast keyword matcher. Tries direct label match and number-word match.

        Returns: matching candidate dict, OR None if no clear match.

        Examples:
          "bottle 1"     → bottle 1
          "bottle one"   → bottle 1
          "the first"    → bottle 1 (if all candidates are bottles)
          "number two"   → bottle 2
          "broccoli"     → None
        """
        if not candidates:
            return None

        text = reply.lower().strip().rstrip('.!?')

        # ----- Strategy A: Direct label match (e.g. "bottle 1") -----
        for c in candidates:
            label = c.get('label', '').lower()
            if label and label in text:
                return c

        # ----- Strategy B: Word-number match (e.g. "bottle one") -----
        # Build pattern: <class> + <number-word>
        # e.g. "bottle one" → look for class "bottle", number 1
        for c in candidates:
            cls = c.get('class', '').lower()
            if not cls:
                continue
            if cls in text:
                # Find what number follows the class word (or precedes)
                for word, num in self._NUM_WORDS.items():
                    if word in text.split():
                        # Check this candidate's label has this number
                        target_label = f"{cls} {num}"
                        if c.get('label', '').lower() == target_label:
                            return c

        # ----- Strategy C: Plain number-word, single class case -----
        # If all candidates same class, "first" / "two" / "1" alone is enough
        all_classes = {c.get('class', '') for c in candidates}
        if len(all_classes) == 1:
            cls = list(all_classes)[0]
            words = text.split()
            for word in words:
                if word in self._NUM_WORDS:
                    target_num = self._NUM_WORDS[word]
                    target_label = f"{cls} {target_num}"
                    for c in candidates:
                        if c.get('label', '').lower() == target_label:
                            return c
                # Also handle "1", "2" as digits
                if word.isdigit():
                    target_label = f"{cls} {int(word)}"
                    for c in candidates:
                        if c.get('label', '').lower() == target_label:
                            return c

        return None

    def _match_label_llm(self, reply: str, candidates: list):
        """
        LLM-based matcher for natural language replies.
        Used only if keyword matcher fails.

        Examples:
          "the closer one"  → bottle 1 (lower distance)
          "the far one"     → bottle 2 (higher distance)
          "the small one"   → cannot decide → None

        Returns: matching candidate dict, OR None.
        """
        if not candidates:
            return None

        # Build candidate description for LLM
        cand_lines = []
        for c in candidates:
            cand_lines.append(
                f"- {c['label']}: distance={c['distance']:.2f}m, "
                f"reachable={c['reachable']}, pose={c['pose']}"
            )
        cand_block = "\n".join(cand_lines)

        prompt = (
            "You help a robot pick the correct object based on a user's reply.\n"
            "The robot asked the user to choose between these objects:\n"
            f"{cand_block}\n\n"
            f'The user replied: "{reply}"\n\n'
            "Reply with EXACTLY ONE of these labels (no other words, no punctuation):\n"
            f"{', '.join(c['label'] for c in candidates)}\n"
            "OR reply with INVALID if the user did not pick one."
        )

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        # temperature 0.0: deterministic label pick (no creativity
                        # wanted — we need one exact label back).
                        # num_predict 20: reply is just a short label, so cap output
                        "temperature": 0.0,
                        "num_predict": 20,
                    }
                },
                timeout=15,
            )
            if response.status_code != 200:
                return None
            llm_out = response.json().get('response', '').strip().lower()
            # Match against candidate labels
            for c in candidates:
                if c['label'].lower() in llm_out:
                    self.get_logger().info(
                        f' LLM matched "{reply}" → {c["label"]}'
                    )
                    return c
            self.get_logger().info(f' LLM said: "{llm_out}" — no match')
            return None
        except Exception as e:
            self.get_logger().warn(f' LLM matcher error: {e}')
            return None

    def _refresh_candidate_from_scene(self, candidate: dict):
        """
        Re-validate a saved candidate against the LATEST scene.
        Returns: fresh candidate dict (with current world coords), OR None
                 if the object is no longer visible.
        """
        target_label = candidate.get('label')
        for fresh in self.current_objects:
            if fresh.get('label') == target_label:
                return fresh
        return None

    def _build_pick_command(self, obj: dict) -> dict:
        """
        Build the rich /llm/command JSON for a resolved pick.
        Uses world coordinates and metadata from the chosen object.
        """
        return {
            "action":                 "pick",
            "target_label":           obj.get('label'),
            "object_class":           obj.get('class'),
            "pose":                   obj.get('pose'),
            "world_x":                obj.get('world_x'),
            "world_y":                obj.get('world_y'),
            "world_z":                obj.get('world_z'),
            "target_bin":             None,    # arm controller fills if needed
            "feedback":               f"Picking {obj.get('label')}.",
            "reasoning":              f"User chose {obj.get('label')} "
                                      f"at ({obj.get('world_x'):.2f}, "
                                      f"{obj.get('world_y'):.2f}, "
                                      f"{obj.get('world_z'):.2f}).",
            "safe":                   bool(obj.get('reachable', False)),
            "clarification_needed":   False,
            "clarification_question": None,
            # Legacy fields
            "object":                 obj.get('class'),
        }

    def _build_unsafe_pick_command(self, obj: dict, reason: str) -> dict:
        """ Build a rejected pick command (e.g. unreachable target)."""
        return {
            "action":                 "error",
            "target_label":           obj.get('label'),
            "object_class":           obj.get('class'),
            "pose":                   obj.get('pose'),
            "world_x":                obj.get('world_x'),
            "world_y":                obj.get('world_y'),
            "world_z":                obj.get('world_z'),
            "target_bin":             None,
            "feedback":               reason,
            "reasoning":              reason,
            "safe":                   False,
            "clarification_needed":   False,
            "clarification_question": None,
            "object":                 obj.get('class'),
        }


    # CLARIFICATION HANDLER

    def handle_clarification_response(self, user_text: str) -> str:
        """
        Resolve a clarification reply to a specific object:
          1. Try keyword matcher (fast).
          2. Fallback to LLM matcher (handles natural language).
          3. Re-validate the chosen object against latest scene.
          4. Build rich pick command with world coordinates.
          5. On invalid reply: re-ask with hint (max MAX_RETRIES tries).
        """
        candidates = self.pending_candidates
        labels = [c.get('label', '?') for c in candidates]
        self.get_logger().info(
            f' Clarification reply: "{user_text}" '
            f'for candidates {labels}'
        )

        # --- Step 1: Try keyword matcher (fast) ---
        matched = self._match_label_keywords(user_text, candidates)
        if matched:
            self.get_logger().info(
                f' Keyword matcher: "{user_text}" → {matched["label"]}'
            )
        else:
            # --- Step 2: Fallback to LLM matcher (slower, smarter) ---
            self.get_logger().info(
                f' Keyword matcher failed → trying LLM matcher'
            )
            matched = self._match_label_llm(user_text, candidates)

        # --- Step 3: Handle invalid reply (re-ask with hint) ---
        if matched is None:
            self.pending_retries += 1
            if self.pending_retries < self.MAX_RETRIES:
                hint = ", ".join(labels[:-1]) + f", or {labels[-1]}" \
                    if len(labels) > 1 else labels[0]
                feedback = (
                    f"I didn't understand. Please say {hint}. "
                    f"(or say 'cancel' to stop)"
                )
                self.get_logger().info(
                    f' Invalid reply (retry {self.pending_retries}/'
                    f'{self.MAX_RETRIES}) — re-asking'
                )
                # Stay in AWAITING_RESPONSE
                return json.dumps({
                    "action": "clarify",
                    "target_label": None,
                    "object_class": None,
                    "pose": None,
                    "world_x": None, "world_y": None, "world_z": None,
                    "target_bin": None,
                    "feedback": feedback,
                    "reasoning": "User reply did not match any candidate.",
                    "safe": True,
                    "clarification_needed": True,
                    "clarification_question": feedback,
                    "object": None,
                })
            else:
                # Max retries reached → give up
                self.get_logger().info(' Max retries reached — giving up')
                self._clear_state(reason="max retries reached")
                return json.dumps({
                    "action": "error",
                    "target_label": None,
                    "object_class": None,
                    "pose": None,
                    "world_x": None, "world_y": None, "world_z": None,
                    "target_bin": None,
                    "feedback": "I'm having trouble understanding. "
                                "Please try your command again.",
                    "reasoning": "Failed to match clarification reply "
                                 f"after {self.MAX_RETRIES} attempts.",
                    "safe": True,
                    "clarification_needed": False,
                    "clarification_question": None,
                    "object": None,
                })

        # --- Step 4: Re-validate against latest scene ---
        fresh = self._refresh_candidate_from_scene(matched)
        if fresh is None:
            # Object disappeared from scene
            self.get_logger().warn(
                f' Object {matched["label"]} no longer in scene — aborting'
            )
            self._clear_state(reason="object disappeared")
            return json.dumps({
                "action": "error",
                "target_label": matched.get('label'),
                "object_class": matched.get('class'),
                "pose": None,
                "world_x": None, "world_y": None, "world_z": None,
                "target_bin": None,
                "feedback": f"I no longer see {matched['label']}. "
                            "Please try again.",
                "reasoning": "Chosen object missing from latest scene.",
                "safe": False,
                "clarification_needed": False,
                "clarification_question": None,
                "object": matched.get('class'),
            })

        # --- Step 5: Build rich pick command ---
        if not fresh.get('reachable', False):
            # Object exists but is too far
            self._clear_state(reason="object unreachable")
            cmd = self._build_unsafe_pick_command(
                fresh,
                f"{fresh['label']} is too far to reach "
                f"(distance: {fresh['distance']:.2f}m). I cannot pick it."
            )
            return json.dumps(cmd)

        # Success: reset state, build command
        self._clear_state(reason="clarification resolved")
        cmd = self._build_pick_command(fresh)
        self.get_logger().info(
            f' Resolved pick: {fresh["label"]} at '
            f'({fresh["world_x"]:.2f}, {fresh["world_y"]:.2f}, '
            f'{fresh["world_z"]:.2f})'
        )
        return json.dumps(cmd)

    # ==============================================================
    # LLM PROCESSING
    # ==============================================================

    def process_with_llm(self, user_text: str) -> str:
        # --- Step 1: Classify action type ---
        action_type = self.knowledge.classify_action(user_text)
        self.get_logger().info(f' Pre-classified action: {action_type}')

        # --- Step 1.5: Direct spatial query ---
        # Handle "what is the closest bottle?" / "which can is farthest?"
        # without going to the LLM.
        if action_type == "query" and self.current_objects:
            spatial_intent = self._detect_spatial_intent(user_text)
            if spatial_intent is not None:
                target_class = self._extract_target_class(user_text)
                pool = (self._candidates_for_class(target_class)
                        if target_class else list(self.current_objects))
                if not pool:
                    feedback = (f"I don't see any {target_class}s on the table."
                                if target_class
                                else "I don't see any objects on the table.")
                    return json.dumps({
                        "action": "query",
                        "object": target_class,
                        "target_bin": None,
                        "feedback": feedback,
                        "reasoning": "No matching objects.",
                        "clarification_needed": False,
                        "clarification_question": None,
                        "safe": True,
                    })

                chosen = self._resolve_spatial(pool, spatial_intent)
                spatial_word = "closest" if spatial_intent == "near" else "farthest"
                feedback = (
                    f"The {spatial_word} {target_class or 'object'} is "
                    f"{chosen['label']} at {chosen['distance']:.2f}m."
                )
                self.get_logger().info(
                    f' Spatial query answered: {chosen["label"]} ({spatial_word})'
                )
                return json.dumps({
                    "action": "query",
                    "object": target_class,
                    "target_bin": None,
                    "feedback": feedback,
                    "reasoning": f"Identified {spatial_word} matching object.",
                    "clarification_needed": False,
                    "clarification_question": None,
                    "safe": True,
                })

        # --- Step 2a: Out-of-scope guard ---
        if action_type == "unknown":
            self.get_logger().info(' Out-of-scope command — rejecting')
            return json.dumps({
                "action": "error",
                "object": None,
                "target_bin": None,
                "feedback": "I can only help with sorting bottles and cans. "
                            "You can ask me to pick, sort, or describe what I see on the table.",
                "reasoning": "Command is not related to Pfand sorting tasks.",
                "clarification_needed": False,
                "clarification_question": None,
                "safe": True
            })

        # --- Step 2b: Vision staleness check ---
        if self.current_scene is not None and self.vision_last_update is not None:
            elapsed = (self.get_clock().now() - self.vision_last_update).nanoseconds / 1e9
            if elapsed > self.vision_timeout:
                self.get_logger().warn(
                    f'  Vision data is {elapsed:.1f}s old (timeout: {self.vision_timeout}s) — marking stale')
                self.current_scene = None
                self.current_objects = []
                self.is_ambiguous = False
                self.ambiguous_classes = []
                self.scene_description = ""
                self.vision_last_update = None

        # --- Step 2c: Vision guard ---
        if self.current_scene is None and action_type != "unknown":
            self.get_logger().warn('  No vision data — cannot proceed')
            return json.dumps({
                "action": "error",
                "object": None,
                "target_bin": None,
                "feedback": "I cannot see the table right now. Please check the camera.",
                "reasoning": "No vision data available.",
                "clarification_needed": False,
                "clarification_question": None,
                "safe": False
            })

        # --- Step 2d: SMART AMBIGUITY RESOLVER
        # Decide what to do based on:
        #   - Explicit label ("bottle 1", "can 2")
        #   - Spatial intent ("closer"/"farther")
        #   - User-mentioned class (bottle/can/none)
        #   - Scene contents (current_objects from Part 1)
        if action_type in ("pick", "place") and self.current_objects:
            target_class    = self._extract_target_class(user_text)
            spatial_intent  = self._detect_spatial_intent(user_text)

            # ----- Explicit label ("pick the bottle 1") -----
            # If user directly named an object (e.g. "bottle 1", "can 2"),
            # skip clarification entirely and go straight to pick.
            direct = self._match_label_keywords(user_text, self.current_objects)
            if direct is not None:
                self.get_logger().info(
                    f' Direct label match: "{user_text}" → {direct["label"]}'
                )
                if not direct.get('reachable', False):
                    return json.dumps(self._build_unsafe_pick_command(
                        direct,
                        f"{direct['label']} is too far to reach "
                        f"(distance: {direct['distance']:.2f}m). "
                        "I cannot pick it."
                    ))
                return json.dumps(self._build_pick_command(direct))

            # ----- Spatial intent ("the closer one") -----
            if spatial_intent is not None:
                # Filter by class if user gave one, else use all objects
                pool = (self._candidates_for_class(target_class)
                        if target_class else list(self.current_objects))
                if not pool:
                    return json.dumps({
                        "action": "error",
                        "object": target_class,
                        "target_bin": None,
                        "feedback": (f"I don't see any {target_class}s on the table."
                                     if target_class
                                     else "I don't see any objects to pick."),
                        "reasoning": "No matching objects in scene.",
                        "clarification_needed": False,
                        "clarification_question": None,
                        "safe": True,
                    })

                chosen = self._resolve_spatial(pool, spatial_intent)
                self.get_logger().info(
                    f' Spatial auto-pick: intent="{spatial_intent}" → '
                    f'{chosen["label"]}'
                )

                if not chosen.get('reachable', False):
                    return json.dumps(self._build_unsafe_pick_command(
                        chosen,
                        f"{chosen['label']} is too far to reach "
                        f"(distance: {chosen['distance']:.2f}m). "
                        "I cannot pick it."
                    ))
                # Add a small note about why we chose it
                cmd = self._build_pick_command(chosen)
                spatial_word = "nearest" if spatial_intent == "near" else "farthest"
                cmd["feedback"] = (
                    f"Picking {chosen['label']} "
                    f"(the {spatial_word} one at {chosen['distance']:.2f}m)."
                )
                return json.dumps(cmd)

            # ----- CASE A: User mentioned a class (e.g. "pick the bottle") -----
            if target_class is not None:
                candidates = self._candidates_for_class(target_class)
                n = len(candidates)

                if n == 0:
                    # No object of that class on table
                    return json.dumps({
                        "action": "error",
                        "object": target_class,
                        "target_bin": None,
                        "feedback": f"I don't see any {target_class}s on the table.",
                        "reasoning": f"No {target_class} detected in current scene.",
                        "clarification_needed": False,
                        "clarification_question": None,
                        "safe": True,
                    })

                elif n == 1:
                    # Only one — auto-resolve directly to a pick command
                    only = candidates[0]
                    self.get_logger().info(
                        f' Auto-resolved: only one {target_class} → '
                        f'{only.get("label")}'
                    )
                    if not only.get('reachable', False):
                        return json.dumps(self._build_unsafe_pick_command(
                            only,
                            f"{only['label']} is too far to reach "
                            f"(distance: {only['distance']:.2f}m). "
                            "I cannot pick it."
                        ))
                    return json.dumps(self._build_pick_command(only))

                else:
                    # MULTIPLE same-class — ASK clarification
                    question = self._format_same_class_question(
                        target_class, candidates
                    )
                    self._set_awaiting(
                        action=action_type,
                        object_class=target_class,
                        candidates=candidates,
                        question=question,
                    )
                    self.get_logger().info(
                        f' Same-class ambiguity ({n} {target_class}s) — asking'
                    )
                    return json.dumps({
                        "action": "clarify",
                        "object": target_class,
                        "target_bin": None,
                        "feedback": question,
                        "reasoning": f"Multiple {target_class}s visible — need user choice.",
                        "clarification_needed": True,
                        "clarification_question": question,
                        "safe": True,
                    })

            # ----- CASE B: No class mentioned + pronoun + multiple objects -----
            # Fallback: keep the old pronoun guard for cases like "pick it"
            elif self._has_pronoun(user_text) and len(self.current_objects) > 1:
                # Build a generic question with class options
                classes_present = sorted({obj['class'] for obj in self.current_objects})
                if len(classes_present) == 1:
                    # All same class but no class mentioned in command
                    cls = classes_present[0]
                    question = self._format_same_class_question(
                        cls, self.current_objects
                    )
                    candidates_for_state = self.current_objects
                    obj_for_state = cls
                else:
                    # Mixed classes — ask which type first
                    options = " or ".join(f"a {c}" for c in classes_present)
                    question = (
                        f"I see {self.scene_description.replace('I see ', '', 1) if self.scene_description.startswith('I see ') else self.scene_description} "
                        f"Do you mean {options}?"
                    )
                    candidates_for_state = self.current_objects
                    obj_for_state = None

                self._set_awaiting(
                    action=action_type,
                    object_class=obj_for_state,
                    candidates=candidates_for_state,
                    question=question,
                )
                self.get_logger().info(' Pronoun + multiple objects — asking')
                return json.dumps({
                    "action": "clarify",
                    "object": None,
                    "target_bin": None,
                    "feedback": question,
                    "reasoning": "User used pronoun with multiple objects visible.",
                    "clarification_needed": True,
                    "clarification_question": question,
                    "safe": True,
                })

        # --- Step 3: Selective context loading ---
        relevant_context = self.knowledge.get_relevant_context(
            action_type, user_text
        )

        # --- Step 4: Build vision context ---
        if self.current_scene:
            vision_context = (
                f"Current Scene (live camera):\n"
                f"- Objects detected: {self.current_scene['objects']}\n"
                f"- Description: {self.current_scene['description']}"
            )
        else:
            vision_context = "Current Scene: No vision data available yet."

        # --- Step 5: Assemble full prompt ---
        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"--- RELEVANT RULES (apply these) ---\n"
            f"{relevant_context}\n\n"
            f"--- SCENE ---\n"
            f"{vision_context}\n\n"
            f"User command: \"{user_text}\"\n\n"
            f"Apply all relevant rules above and generate your JSON action plan:"
        )

        approx_tokens = len(full_prompt.split()) * 1.3
        self.get_logger().info(f' Prompt size: ~{int(approx_tokens)} tokens')

        # --- Step 6: Call Ollama ---
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        # temperature 0.1: near-deterministic but allows slight
                        # phrasing variation in the feedback text.
                        # num_predict 250: enough for the JSON action plan.
                        "temperature": 0.1,
                        "num_predict": 250
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                llm_output = result.get('response', '').strip()

                json_output = self.extract_json(llm_output)
                if json_output:
                    return json.dumps(json_output)
                else:
                    self.get_logger().warn(f'  Could not parse JSON from LLM:\n{llm_output[:300]}')
                    return self.error_json('invalid_json')
            else:
                return self.error_json('service_error')

        except Exception as e:
            self.get_logger().error(f'LLM error: {e}')
            return self.error_json('exception')

    # ==============================================================
    # HELPERS
    # ==============================================================

    def extract_json(self, text: str):
        """Extract the first JSON object from LLM output."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def generate_feedback(self, response_data: dict) -> str:
        """Generate natural-language feedback from the LLM response."""
        if response_data.get("feedback"):
            return response_data["feedback"]

        if response_data.get('clarification_needed'):
            return response_data.get('clarification_question',
                                     'Could you please clarify?')

        if response_data.get('safe') is False:
            return response_data.get('reasoning',
                                     'That action is not safe to perform.')

        action = response_data.get('action', '')
        if action == 'query':
            if self.current_scene:
                return f"I see {self.current_scene['description']} on the table."
            return "I cannot see the table right now."

        if action == 'count':
            obj_type = response_data.get('object', 'objects')
            if self.current_scene:
                count = sum(
                    1 for obj in self.current_scene['objects']
                    if obj_type == 'all' or obj == obj_type
                )
                return f"There are {count} {obj_type} on the table."
            return "I cannot count objects without vision."

        if action in ('sort', 'pick', 'place'):
            obj = response_data.get('object', 'object')
            bin_target = response_data.get('target_bin', '')
            reasoning = response_data.get('reasoning', '')
            if reasoning:
                return reasoning
            return f"{'Sorting' if action == 'sort' else 'Picking'} {obj}" + \
                   (f" to Bin {bin_target}." if bin_target else ".")

        return response_data.get('reasoning', 'Command received.')

    def error_json(self, error_type: str) -> str:
        """Generate a structured error response."""
        return json.dumps({
            "action": "error",
            "error_type": error_type,
            "feedback": "I encountered an error processing your command. Please try again.",
            "safe": False,
            "clarification_needed": False
        })


# ==================================================================
# ENTRY POINT
# ==================================================================

def main(args=None):
    rclpy.init(args=args)
    node = EnhancedLLMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
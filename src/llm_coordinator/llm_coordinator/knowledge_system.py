#!/usr/bin/env python3
"""
Lightweight Knowledge System with Selective Context Loading

Loads all knowledge files once at startup, then returns ONLY the
relevant context based on the detected action type and user query.

No vector databases, no embeddings — just fast IF-THEN selection.
Retrieval target: <10ms, ~200-400 tokens per query.
"""

import os
import time
import yaml


class LightweightKnowledge:
    """
    Selective context loader for the Pfand sorting LLM coordinator.

    Usage:
        knowledge = LightweightKnowledge(
            knowledge_path="/path/to/knowledge_base",
            config_path="/path/to/config/robot_config.yaml"
        )
        context = knowledge.get_relevant_context(action_type, user_query)
    """

    # Words that signal the user is being ambiguous
    AMBIGUOUS_PRONOUNS = {"it", "that", "this", "them", "those", "the one"}

    # Words that signal a "sort everything" / bulk operation
    BULK_KEYWORDS = {"everything", "all", "clear", "table", "sort all",
                     "clear the table", "sort everything"}

    def __init__(self, knowledge_path: str, config_path: str):
        """
        Load all knowledge files and robot config once at startup.

        Args:
            knowledge_path: Directory containing the 5 .txt knowledge files.
            config_path:    Path to robot_config.yaml.
        """
        self.knowledge = {}
        self.robot_config = {}

        # --- Load knowledge files ---
        knowledge_files = {
            "sorting":        "sorting_rules.txt",
            "safety":         "safety_rules.txt",
            "priority":       "priority_rules.txt",
            "clarification":  "clarification_rules.txt",
            "feedback":       "feedback_templates.txt",
        }

        for key, filename in knowledge_files.items():
            filepath = os.path.join(knowledge_path, filename)
            try:
                with open(filepath, "r") as f:
                    self.knowledge[key] = f.read()
            except FileNotFoundError:
                print(f"[Knowledge] WARNING: {filepath} not found — skipping.")
                self.knowledge[key] = ""

        # --- Load robot config ---
        try:
            with open(config_path, "r") as f:
                self.robot_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[Knowledge] WARNING: {config_path} not found — using empty config.")
            self.robot_config = {}

        # --- Pre-format config summaries (done once, reused every query) ---
        self._workspace_summary = self._build_workspace_summary()
        self._bins_summary = self._build_bins_summary()

        print(f"[Knowledge] Loaded {len(self.knowledge)} knowledge files.")
        print(f"[Knowledge] Robot config: {self.robot_config.get('robot', {}).get('name', 'unknown')}")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def get_relevant_context(self, action_type: str, user_query: str) -> str:
        """
        Select and return only the knowledge relevant to this query.

        Args:
            action_type: One of "pick", "place", "sort", "count", "query", "unknown".
            user_query:  The raw user command string.

        Returns:
            A single string with selected knowledge + config, ready to
            inject into the LLM prompt.  Target: ~200-400 tokens.
        """
        start = time.perf_counter_ns()

        sections = []
        query_lower = user_query.lower()

        # ----- 1. Safety rules: needed for any physical action -----
        if action_type in ("pick", "place", "sort"):
            sections.append(("Safety Rules", self.knowledge["safety"]))

        # ----- 2. Sorting rules: needed when sorting -----
        if action_type == "sort":
            sections.append(("Sorting Rules", self.knowledge["sorting"]))

        # ----- 3. Priority rules: needed for bulk / multi-object -----
        if action_type == "sort" and self._has_bulk_keyword(query_lower):
            sections.append(("Priority Rules", self.knowledge["priority"]))

        # ----- 4. Clarification rules: needed when query is ambiguous -----
        if self._is_ambiguous(query_lower):
            sections.append(("Clarification Rules", self.knowledge["clarification"]))

        # ----- 5. Feedback templates: always include (lightweight) -----
        sections.append(("Feedback Templates", self.knowledge["feedback"]))

        # ----- 6. Robot config: workspace + bins for physical actions -----
        config_block = ""
        if action_type in ("pick", "place", "sort"):
            config_block = self._workspace_summary + "\n" + self._bins_summary

        # ----- Assemble final context string -----
        context_parts = []
        for title, content in sections:
            # Strip comment-only header lines to save tokens
            trimmed = self._strip_header_comments(content)
            context_parts.append(f"### {title}\n{trimmed}")

        if config_block:
            context_parts.append(f"### Robot Configuration\n{config_block}")

        context = "\n\n".join(context_parts)

        elapsed_us = (time.perf_counter_ns() - start) / 1_000
        print(f"[Knowledge] Selected {len(sections)} sections in {elapsed_us:.0f} µs  "
              f"(~{len(context.split())} words)")

        return context

    def classify_action(self, user_query: str) -> str:
        """
        Classify the user's intent into an action type.

        This is a simple keyword-based classifier.  The LLM also does its
        own classification, but this pre-classification is used to decide
        WHICH knowledge to load (before the LLM sees anything).

        Returns one of: "pick", "place", "sort", "count", "query", "unknown".
        """
        q = user_query.lower()

        # Sort / clear
        if any(w in q for w in ("sort", "clear", "clean", "tidy")):
            return "sort"

        # Pick
        if any(w in q for w in ("pick", "grab", "get", "take", "lift")):
            return "pick"

        # Place / put
        if any(w in q for w in ("place", "put", "move", "drop")):
            return "place"

        # Count
        if any(w in q for w in ("how many", "count", "number of")):
            return "count"

        # Query (describe, what, where, etc.)
        if any(w in q for w in ("what", "where", "describe", "see",
                                 "look", "show", "tell")):
            return "query"

        return "unknown"

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _is_ambiguous(self, query_lower: str) -> bool:
        """Check if the query contains ambiguous pronouns."""
        words = set(query_lower.split())
        return bool(words & self.AMBIGUOUS_PRONOUNS)

    def _has_bulk_keyword(self, query_lower: str) -> bool:
        """Check if the query is asking for a bulk / multi-object operation."""
        return any(kw in query_lower for kw in self.BULK_KEYWORDS)

    def _strip_header_comments(self, text: str) -> str:
        """Remove leading # comment blocks to save tokens."""
        lines = text.strip().splitlines()
        result = []
        past_header = False
        for line in lines:
            if not past_header:
                stripped = line.strip()
                if stripped == "" or stripped.startswith("#"):
                    continue          # skip header comments
                else:
                    past_header = True
            if past_header:
                result.append(line)
        return "\n".join(result)

    def _build_workspace_summary(self) -> str:
        """One-time: compact workspace string from config."""
        ws = self.robot_config.get("workspace", {})
        sf = self.robot_config.get("safety", {})
        if not ws:
            return "Workspace: not configured."
        return (
            f"Workspace limits: "
            f"X=[{ws.get('x_min')}, {ws.get('x_max')}], "
            f"Y=[{ws.get('y_min')}, {ws.get('y_max')}], "
            f"Z=[{ws.get('z_min')}, {ws.get('z_max')}]  "
            f"Edge safety margin: {sf.get('edge_margin', 'N/A')}m"
        )

    def _build_bins_summary(self) -> str:
        """One-time: compact bin positions from config."""
        bins = self.robot_config.get("bins", {})
        if not bins:
            return "Bins: not configured."
        parts = []
        for name, pos in bins.items():
            parts.append(
                f"Bin {name}: ({pos.get('x')}, {pos.get('y')}, {pos.get('z')}) "
                f"- {pos.get('description', '')}"
            )
        return "  |  ".join(parts)


# ======================================================================
# STANDALONE TEST  —  run this file directly to verify knowledge loading
# ======================================================================
if __name__ == "__main__":
    import sys

    # Default paths: go up from llm_coordinator/llm_coordinator/ to package root
    pkg_root = os.path.join(os.path.dirname(__file__), "..")
    kb_path = os.path.join(pkg_root, "knowledge_base")
    cfg_path = os.path.join(pkg_root, "config", "robot_config.yaml")

    if not os.path.isdir(kb_path):
        print(f"Knowledge dir not found at {kb_path}")
        sys.exit(1)

    knowledge = LightweightKnowledge(kb_path, cfg_path)

    # ---- Test cases ----
    test_queries = [
        "Pick the bottle",
        "Sort everything",
        "Pick it up",
        "How many bottles are there?",
        "What's on the table?",
        "Sort them",
        "Clear the table",
    ]

    print("\n" + "=" * 60)
    for query in test_queries:
        action = knowledge.classify_action(query)
        context = knowledge.get_relevant_context(action, query)
        print(f"\nQuery:   \"{query}\"")
        print(f"Action:  {action}")
        print(f"Context: {len(context)} chars  (~{len(context.split())} words)")
        print("-" * 60)
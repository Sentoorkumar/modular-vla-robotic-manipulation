#!/usr/bin/env python3
"""
Test Script: LLM + Knowledge System Integration
Tests if the LLM correctly applies rules from the knowledge base.
No ROS2 needed — just Ollama + knowledge files.
"""

import sys
import os
import json
import requests
import time

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llm_coordinator"))
from knowledge_system import LightweightKnowledge

# ---- Configuration ----
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct-q4_0"

SYSTEM_PROMPT = """You are a Pfand sorting robot coordinator with real-time vision and a knowledge base.

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
- ALWAYS check the Safety Rules before any pick/place/sort action.
- ALWAYS use the Sorting Rules to decide which bin.
- ALWAYS use the Clarification Rules when the command is ambiguous.
- Use Feedback Templates to phrase your spoken response.
- If an action is UNSAFE, set "safe": false and explain in "feedback".
- Output ONLY JSON. No preamble, no markdown, no text outside JSON.
"""

# ---- Test Cases ----
TEST_CASES = [
    {
        "name": "TEST 1: Sorting Rule — bottles to Bin A",
        "query": "Sort the bottles",
        "vision": {"objects": ["bottle", "bottle", "can"], "description": "2 bottles and 1 can"},
        "expect": "Should assign bottles to Bin A",
    },
    {
        "name": "TEST 2: Sorting Rule — cans to Bin B",
        "query": "Sort the cans",
        "vision": {"objects": ["bottle", "can", "can"], "description": "1 bottle and 2 cans"},
        "expect": "Should assign cans to Bin B",
    },
    {
        "name": "TEST 3: Clarification — ambiguous 'it'",
        "query": "Pick it up",
        "vision": {"objects": ["bottle", "can"], "description": "1 bottle and 1 can"},
        "expect": "Should ask which object the user means",
    },
    {
        "name": "TEST 4: Safety — object near edge",
        "query": "Pick the bottle near the edge",
        "vision": {"objects": ["bottle"], "description": "1 bottle near workspace edge"},
        "expect": "Should warn about safety or reject",
    },
    {
        "name": "TEST 5: Sort everything — priority rules",
        "query": "Sort everything",
        "vision": {"objects": ["bottle", "can", "bottle"], "description": "2 bottles and 1 can"},
        "expect": "Should sort bottles first then cans (priority rule)",
    },
    {
        "name": "TEST 6: Query — what's on the table",
        "query": "What do you see?",
        "vision": {"objects": ["bottle", "can", "can"], "description": "1 bottle and 2 cans"},
        "expect": "Should describe the scene",
    },
]


def call_ollama(prompt: str) -> str:
    """Send prompt to Ollama and return raw response."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 250},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        else:
            return f"ERROR: HTTP {resp.status_code}"
    except Exception as e:
        return f"ERROR: {e}"


def main():
    # Load knowledge system
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "robot_config.yaml")

    if not os.path.isdir(kb_path):
        print(f"ERROR: knowledge_base not found at {kb_path}")
        sys.exit(1)

    knowledge = LightweightKnowledge(kb_path, cfg_path)
    print("=" * 70)
    print("  LLM + KNOWLEDGE INTEGRATION TEST")
    print("=" * 70)

    passed = 0
    total = len(TEST_CASES)

    for tc in TEST_CASES:
        print(f"\n{'─' * 70}")
        print(f"  {tc['name']}")
        print(f"  Query:  \"{tc['query']}\"")
        print(f"  Vision: {tc['vision']['description']}")
        print(f"  Expect: {tc['expect']}")
        print(f"{'─' * 70}")

        # Step 1: Classify action
        action_type = knowledge.classify_action(tc["query"])

        # Step 2: Get relevant context
        context = knowledge.get_relevant_context(action_type, tc["query"])

        # Step 3: Build prompt
        vision_text = (
            f"Current Scene (live camera):\n"
            f"- Objects detected: {tc['vision']['objects']}\n"
            f"- Description: {tc['vision']['description']}"
        )

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- RELEVANT RULES (apply these) ---\n"
            f"{context}\n\n"
            f"--- SCENE ---\n"
            f"{vision_text}\n\n"
            f"User command: \"{tc['query']}\"\n\n"
            f"Apply all relevant rules above and generate your JSON action plan:"
        )

        # Step 4: Call LLM
        start = time.time()
        raw_output = call_ollama(prompt)
        elapsed = time.time() - start

        print(f"\n  Action type:    {action_type}")
        print(f"  Response time:  {elapsed:.2f}s")
        print(f"  Raw LLM output:\n")

        # Try to pretty-print JSON
        try:
            parsed = json.loads(raw_output)
            print(f"  {json.dumps(parsed, indent=2)}")
            print(f"\n  ✅ Valid JSON returned")
            passed += 1
        except json.JSONDecodeError:
            # Try extracting JSON from text
            import re
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    print(f"  {json.dumps(parsed, indent=2)}")
                    print(f"\n  ✅ Valid JSON (extracted from text)")
                    passed += 1
                except:
                    print(f"  {raw_output[:500]}")
                    print(f"\n  ❌ Could not parse JSON")
            else:
                print(f"  {raw_output[:500]}")
                print(f"\n  ❌ No JSON found in output")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed}/{total} tests returned valid JSON")
    print(f"{'=' * 70}")
    print(f"\n  Review the outputs above to check if the LLM:")
    print(f"  - Applied sorting rules (bottles→A, cans→B)")
    print(f"  - Asked for clarification on ambiguous commands")
    print(f"  - Warned about safety for edge objects")
    print(f"  - Used priority rules for 'sort everything'")
    print(f"  - Described the scene for query commands")


if __name__ == "__main__":
    main()

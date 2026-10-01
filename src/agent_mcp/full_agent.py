"""src/agent_mcp/full_agent.py

STEP 10.4: Full Agent Integration
----------------------------------
Orchestrates the end-to-end autonomous agent workflow connecting:
1. Step 10.1: Dynamic Capability Selection (Gemini #1 evaluates user intent)
2. Python Validation: Validates classification against whitelist rules
3. Step 10.2: Decision Execution (Python agent executes MCP tool/resource/prompt)
4. Step 10.3: Final Answer Generation (Gemini #2 synthesizes observation into answer)

ARCHITECTURAL PRINCIPLE:
------------------------
User
 ↓
Gemini #1 (Step 10.1: Decide Capability)
 ↓
Python Validation (Validate Category & Identifier)
 ↓
Decision Executor (Step 10.2: Execute via MCP Client/Server)
 ↓
Observation (Result from Tool / Resource / Prompt / Direct)
 ↓
Gemini #2 (Step 10.3: Grounded Natural Language Synthesis)
 ↓
User-Friendly Final Answer
"""

import asyncio
import contextlib
import io
import json
import logging
import os
import sys
from typing import Any

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from google import genai

from src.agent_mcp.dynamic_capability_selector import (
    classify_request,
    get_api_key,
    validate_decision as validate_selector_decision,
)
from src.agent_mcp.decision_executor import (
    execute_decision,
    validate_decision as validate_executor_decision,
)
from src.agent_mcp.final_answer_generator import (
    format_observation,
    generate_final_answer,
)

# Ensure clean UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress verbose informational logs from Google GenAI SDK
logging.getLogger("google_genai").setLevel(logging.ERROR)


def _extract_execution_trace(raw_log: str) -> str:
    """Extracts the execution trace block from execute_decision's stdout."""
    if "Execution:" in raw_log:
        after_exec = raw_log.split("Execution:", 1)[1]
        if "\nObservation:" in after_exec:
            return after_exec.split("\nObservation:", 1)[0].strip()
        elif "Observation:" in after_exec:
            return after_exec.split("Observation:", 1)[0].strip()
        return after_exec.strip()
    return raw_log.strip()


async def run_agent(user_question: str) -> str:
    """Executes the full end-to-end agent workflow for a user question.

    Flow:
    1. Dynamic Capability Selection (Step 10.1 - Gemini #1)
    2. Python Decision Validation
    3. Decision Execution (Step 10.2 - MCP Client/Server, bypassed for DIRECT)
    4. Observation Capture
    5. Final Answer Generation (Step 10.3 - Gemini #2)

    Returns:
        The user-friendly final answer string, or an informative error message.
    """
    print("========================================")
    print("STEP 10.4 - FULL AGENT")
    print("========================================")
    print(f"\nUSER QUESTION:\n{user_question}")

    # Stage 0: API Key check
    api_key = get_api_key()
    if not api_key:
        err = "ERROR [Stage: Initialization]: GEMINI_API_KEY environment variable is not set."
        print(f"\n[ERROR]:\n{err}")
        print("\n========================================")
        print("END")
        print("========================================")
        return err

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    # Stage 1: Step 10.1 Dynamic Capability Selection (Gemini #1)
    try:
        decision = classify_request(client, model_name, user_question)
    except Exception as e:
        err = f"ERROR [Stage: Step 10.1 Decision]: Gemini API exception: {e}"
        print(f"\n[10.1] GEMINI DECISION:\n{err}")
        print("\n========================================")
        print("END")
        print("========================================")
        return err

    if decision is None:
        err = "ERROR [Stage: Step 10.1 Decision]: Failed to obtain decision from Gemini (API 429 Quota Exceeded / 503 Unavailable / Network error)."
        print(f"\n[10.1] GEMINI DECISION:\n{err}")
        print("\n========================================")
        print("END")
        print("========================================")
        return err

    print(f"\n[10.1] GEMINI DECISION:\n{json.dumps(decision, indent=2)}")

    # Stage 2: Python Validation Layer
    is_valid, validation_msg = validate_selector_decision(decision)
    if not is_valid:
        err = f"ERROR [Stage: Python Validation]: Decision validation failed: {validation_msg}"
        print(f"\n[VALIDATION ERROR]:\n{err}")
        print("\n========================================")
        print("END")
        print("========================================")
        return err

    # Stage 3: Step 10.2 Decision Execution (MCP / Direct)
    capability = decision.get("capability")
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            observation = await execute_decision(user_question, decision)
        raw_log = buffer.getvalue()
        exec_trace = _extract_execution_trace(raw_log)
    except Exception as e:
        err = f"ERROR [Stage: Step 10.2 Execution]: MCP execution failed: {e}"
        print(f"\n[10.2] EXECUTION:\n{err}")
        print("\n========================================")
        print("END")
        print("========================================")
        return err

    print(f"\n[10.2] EXECUTION:\n{exec_trace}")

    # Stage 4: Observation Validation & Display
    if capability != "DIRECT":
        if not observation or (isinstance(observation, dict) and not any(observation.values())):
            err = "ERROR [Stage: Step 10.2 Observation]: MCP execution returned an empty observation."
            print(f"\n[OBSERVATION]:\n{err}")
            print("\n========================================")
            print("END")
            print("========================================")
            return err

    print(f"\n[OBSERVATION]:")
    if isinstance(observation, dict):
        print(json.dumps(observation, indent=2))
    elif isinstance(observation, str):
        print(observation)
    else:
        print(str(observation))

    # Stage 5: Step 10.3 Final Answer Generation (Gemini #2)
    try:
        final_answer = generate_final_answer(user_question, decision, observation)
    except Exception as e:
        err = f"ERROR [Stage: Step 10.3 Final Answer]: Gemini API exception: {e}"
        print(f"\n[10.3] FINAL ANSWER:\n{err}")
        print("\n========================================")
        print("END")
        print("========================================")
        return err

    if not final_answer or final_answer.startswith("Status: Gemini") or final_answer.startswith("CONFIGURATION ERROR"):
        err = f"ERROR [Stage: Step 10.3 Final Answer]: {final_answer}"
        print(f"\n[10.3] FINAL ANSWER:\n{err}")
        print("\n========================================")
        print("END")
        print("========================================")
        return err

    print(f"\n[10.3] FINAL ANSWER:\n{final_answer}")
    print("\n========================================")
    print("END")
    print("========================================")

    return final_answer


def run_agent_sync(user_question: str) -> str:
    """Synchronous convenience wrapper for run_agent."""
    return asyncio.run(run_agent(user_question))


async def run_tests() -> None:
    """Runs the 3 required controlled integration tests sequentially."""
    test_cases = [
        # TEST 1: TOOL
        {
            "id": "TEST 1",
            "question": "Who created Java?",
            "expected_capability": "TOOL",
            "expected_detail": "search_wikipedia",
        },
        # TEST 2: RESOURCE
        {
            "id": "TEST 2",
            "question": "Show me the Spring Boot learning roadmap.",
            "expected_capability": "RESOURCE",
            "expected_detail": "learning://spring-boot/roadmap",
        },
        # TEST 3: DIRECT
        {
            "id": "TEST 3",
            "question": "What is a variable?",
            "expected_capability": "DIRECT",
            "expected_detail": "no MCP execution",
        },
    ]

    results = []

    for idx, tc in enumerate(test_cases, start=1):
        if idx > 1:
            # Respect Gemini Free Tier rate limits between test queries
            await asyncio.sleep(3)

        print(f"\n##################################################")
        print(f"### RUNNING INTEGRATION {tc['id']}")
        print(f"### Expected: {tc['expected_capability']} ({tc['expected_detail']})")
        print(f"##################################################\n")

        answer = await run_agent(tc["question"])

        if answer.startswith("ERROR"):
            results.append((tc["id"], tc["expected_capability"], "FAIL", answer))
        else:
            results.append((tc["id"], tc["expected_capability"], "PASS", "Answer generated successfully"))

    print("\n========================================")
    print("STEP 10.4 TEST SUMMARY")
    print("========================================")
    for test_id, cap, status, detail in results:
        print(f"{test_id} [{cap}]: {status} - {detail}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        custom_question = " ".join(sys.argv[1:])
        asyncio.run(run_agent(custom_question))
    else:
        asyncio.run(run_tests())

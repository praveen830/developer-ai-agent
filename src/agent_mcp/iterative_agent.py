"""src/agent_mcp/iterative_agent.py

STEP 11: Full Iterative Agent Loop
-----------------------------------
Implements an autonomous iterative loop where:
1. Gemini decides the initial capability to execute.
2. Python validates and executes it via the MCP Client/Server (Step 10.2).
3. Observation is stored in an accumulated history list.
4. Gemini evaluates the accumulated observations to determine whether enough
   information exists (COMPLETE) or another capability is required (NEEDS_MORE).
5. If more information is needed, Python validates the follow-up capability,
   executes it, and the loop repeats.
6. Once COMPLETE (or safety MAX_ITERATIONS is reached), Gemini synthesizes
   a comprehensive final answer using all accumulated observations (Step 10.3).

ARCHITECTURAL PRINCIPLE:
------------------------
USER
 ↓
GEMINI DECISION (Initial or follow-up)
 ↓
PYTHON VALIDATION (Whitelist enforcement)
 ↓
MCP EXECUTION (Step 10.2 executor)
 ↓
OBSERVATION (Appended to accumulated history)
 ↓
GEMINI EVALUATION
 ↓
ENOUGH INFORMATION?
      ↓ YES ───► FINAL ANSWER (Step 10.3 generator)
      ↓ NO
NEW DECISION ──► PYTHON VALIDATION ──► MCP EXECUTION ──► NEW OBSERVATION ──► ...
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
from google.genai import types
from google.genai.errors import ClientError, ServerError

from src.agent_mcp.dynamic_capability_selector import (
    ALLOWED_CAPABILITIES,
    VALID_CATEGORIES,
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

# Safety limit for maximum iterations
MAX_ITERATIONS = 5

# System prompt for initial capability selection in an iterative workflow
INITIAL_DECISION_PROMPT = """You are an expert MCP Capability Classifier in an iterative agent loop.
Analyze the user request and determine the FIRST capability to execute to begin addressing the user's request.

AVAILABLE CAPABILITIES:
1. TOOLS (Executable Actions / Factual Lookups):
   - search_wikipedia: Retrieve factual/encyclopedic information (e.g. creators, history, definitions).
   - get_learning_topics: Retrieve structured Spring Boot learning topics.
   - get_student_profile: Retrieve the authenticated student's profile from CareerAI (academic details, degree, branch, college, current year, CGPA, graduation year, profile completion). Requires NO arguments (arguments = {}).
   - get_student_skills: Retrieve the authenticated student's technical skill inventory from CareerAI (skill name, proficiency, experience, last used). Requires NO arguments (arguments = {}).
2. RESOURCE (Authoritative Readable Data):
   - learning://spring-boot/roadmap: Read the authoritative Spring Boot learning roadmap document.
3. PROMPT (Pedagogical Teaching Instructions):
   - spring_boot_teacher: Pedagogical guidelines for teaching concepts to beginners.
4. DIRECT:
   - Basic conceptual questions that do not require external verification or specialized tools (e.g. "What is a variable?").

ITERATIVE AGENT RULES:
- Since this is an iterative agent loop, select ONLY ONE capability for the first step. Do NOT select MULTI.
- For multi-part requests:
  - If a user asks for both student profile/academic information (e.g. current year, CGPA, degree, college) AND technical skills (e.g. "What is my current CGPA and what technical skills do I have?"), select only ONE capability for the current iteration (e.g. "get_student_profile").
  - The remaining capability (e.g. "get_student_skills") will be requested by the evaluator using NEEDS_MORE in the next iteration.
  - For multi-part technical queries (e.g. "Who created Java and give me the Spring Boot roadmap"), pick the first logical capability (such as searching Wikipedia for the creator of Java). Follow-up capabilities will be executed in subsequent iterations.
- If selecting a TOOL, specify the "topic" field if applicable (e.g., "Java" for "Who created Java?"). For zero-argument tools ("get_student_profile", "get_student_skills"), no topic is needed (omit or leave empty, arguments are {}).
- For DIRECT: "recommended_capabilities" must be an empty list [].
- For TOOL, RESOURCE, or PROMPT: "recommended_capabilities" must contain exactly one capability identifier.

You MUST respond ONLY with a strict JSON object following this exact schema:
{
  "capability": "TOOL" | "RESOURCE" | "PROMPT" | "DIRECT",
  "reason": "concise explanation of why this starting capability was chosen",
  "recommended_capabilities": ["single_capability_identifier"],
  "topic": "optional_topic_string"
}
"""

# System prompt for iterative evaluation after observations
EVALUATION_SYSTEM_PROMPT = """You are an expert MCP Agent Evaluator in an iterative loop.
Analyze the user's original request and the accumulated observations gathered so far.
Determine whether enough information exists to thoroughly and accurately answer the user's question.

AVAILABLE MCP CAPABILITIES:
1. TOOLS:
   - search_wikipedia: Retrieve factual/encyclopedic information from Wikipedia.
   - get_learning_topics: Retrieve structured Spring Boot learning topics.
   - get_student_profile: Retrieve the authenticated student's profile from CareerAI (academic details, degree, branch, college, current year, CGPA, graduation year, profile completion). Takes no arguments.
   - get_student_skills: Retrieve the authenticated student's technical skill inventory from CareerAI (skill name, proficiency, experience, last used). Takes no arguments.
2. RESOURCE:
   - learning://spring-boot/roadmap: Read the authoritative Spring Boot learning roadmap.
3. PROMPT:
   - spring_boot_teacher: Pedagogical guidelines for teaching technical concepts to beginners.

EVALUATION RULES:
1. Review the original user request carefully. Did the user ask multiple questions or request multiple resources?
2. You must consider ALL accumulated observations, not only the latest observation.
3. Multi-tool student query rules:
   - If the user requested both student profile/academic information (such as CGPA, year, degree) AND technical skills:
     * If only get_student_profile has been observed, return status "NEEDS_MORE" and request get_student_skills as next_capability.
     * If only get_student_skills has been observed, return status "NEEDS_MORE" and request get_student_profile as next_capability.
     * If both relevant observations are available in the accumulated observations, return status "COMPLETE".
4. If ALL aspects of the user's question can be thoroughly answered using the accumulated observations (or if it was a DIRECT question), return status "COMPLETE".
5. If critical information requested by the user is STILL MISSING (e.g., user asked for Java creator AND Spring Boot roadmap, but only Java creator was retrieved), return status "NEEDS_MORE" and specify the exact next capability to execute.
6. Do NOT request a capability that has already been successfully executed unless there is a clear reason.
7. If requesting a TOOL, specify the "name" and optional "topic" parameter (omit or leave empty for zero-argument tools like "get_student_profile" and "get_student_skills").

You MUST respond ONLY with a strict JSON object following one of these two exact schemas:

If enough information exists:
{
  "status": "COMPLETE",
  "reason": "Detailed explanation of why the gathered information is sufficient.",
  "next_capability": null
}

If more information is needed:
{
  "status": "NEEDS_MORE",
  "reason": "Explanation of what specific information is still missing.",
  "next_capability": {
    "capability": "TOOL" | "RESOURCE" | "PROMPT",
    "name": "capability_identifier",
    "topic": "optional_topic_string"
  }
}
"""


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


def decide_initial_capability(
    client: genai.Client,
    model_name: str,
    user_question: str,
) -> dict[str, Any] | None:
    """Invokes Gemini to select the first single capability for the iterative loop."""
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"{INITIAL_DECISION_PROMPT}\n\nUSER REQUEST:\n\"{user_question}\""
                )
            ],
        )
    ]
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        if response and response.text:
            return json.loads(response.text)
        return None
    except ClientError as e:
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            print(f"GEMINI NOTICE (429 Quota Exceeded in Initial Decision): {e}", file=sys.stderr)
        else:
            print(f"GEMINI CLIENT ERROR ({e.code}) in Initial Decision: {e}", file=sys.stderr)
        return None
    except ServerError as e:
        if e.code == 503 or "UNAVAILABLE" in str(e):
            print(f"GEMINI NOTICE (503 Service Unavailable in Initial Decision): {e}", file=sys.stderr)
        else:
            print(f"GEMINI SERVER ERROR ({e.code}) in Initial Decision: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"GEMINI UNEXPECTED ERROR in Initial Decision: {e}", file=sys.stderr)
        return None


def evaluate_observations(
    client: genai.Client,
    model_name: str,
    user_question: str,
    previous_decisions: list[dict[str, Any]],
    observations: list[Any],
) -> dict[str, Any] | None:
    """Invokes Gemini to evaluate accumulated observations and decide if more capabilities are required."""
    prompt = f"""{EVALUATION_SYSTEM_PROMPT}

ORIGINAL USER REQUEST:
\"{user_question}\"

CAPABILITIES EXECUTED SO FAR:
{json.dumps(previous_decisions, indent=2)}

ACCUMULATED OBSERVATIONS:
{json.dumps(observations, indent=2)}
"""
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        if response and response.text:
            return json.loads(response.text)
        return None
    except ClientError as e:
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            print(f"GEMINI NOTICE (429 Quota Exceeded in Evaluation): {e}", file=sys.stderr)
        else:
            print(f"GEMINI CLIENT ERROR ({e.code}) in Evaluation: {e}", file=sys.stderr)
        return None
    except ServerError as e:
        if e.code == 503 or "UNAVAILABLE" in str(e):
            print(f"GEMINI NOTICE (503 Service Unavailable in Evaluation): {e}", file=sys.stderr)
        else:
            print(f"GEMINI SERVER ERROR ({e.code}) in Evaluation: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"GEMINI UNEXPECTED ERROR in Evaluation: {e}", file=sys.stderr)
        return None


def validate_evaluation(eval_result: Any) -> tuple[bool, str]:
    """Validates the structured evaluation JSON response from Gemini."""
    if not isinstance(eval_result, dict):
        return False, "Evaluation result is not a dictionary."

    status = eval_result.get("status")
    if status not in {"COMPLETE", "NEEDS_MORE"}:
        return False, f"Invalid status '{status}'. Must be 'COMPLETE' or 'NEEDS_MORE'."

    reason = eval_result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False, "Field 'reason' must be a non-empty string."

    if status == "COMPLETE":
        if eval_result.get("next_capability") is not None:
            return False, "COMPLETE status requires next_capability to be null."

    elif status == "NEEDS_MORE":
        next_cap = eval_result.get("next_capability")
        if not isinstance(next_cap, dict):
            return False, "NEEDS_MORE status requires next_capability to be a dictionary."

        cap_type = next_cap.get("capability")
        if cap_type not in {"TOOL", "RESOURCE", "PROMPT"}:
            return False, f"Invalid next capability type '{cap_type}'."

        cap_name = next_cap.get("name")
        if cap_name not in ALLOWED_CAPABILITIES:
            return False, f"Unauthorized capability '{cap_name}'. Allowed: {sorted(ALLOWED_CAPABILITIES)}."

    return True, "PASS"


async def run_iterative_agent(
    user_question: str,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """Executes the full iterative agent loop for a user question.

    Maintains loop state:
    - original user question
    - iteration number
    - previous decisions
    - accumulated observations
    - final answer status

    Returns:
        The final user-friendly answer string, or an informative error message.
    """
    print("========================================")
    print("STEP 11 - ITERATIVE AGENT")
    print("========================================")
    print(f"\nUSER:\n{user_question}")

    # Stage 0: Initialization & API Key retrieval
    api_key = get_api_key()
    if not api_key:
        err = "ERROR [Stage: Initialization]: GEMINI_API_KEY environment variable is not set."
        print(f"\n[ERROR]: {err}")
        print("\n========================================")
        print("END STEP 11")
        print("========================================")
        return err

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    # Loop State
    iteration = 1
    previous_decisions: list[dict[str, Any]] = []
    observations: list[Any] = []
    executed_signatures: set[str] = set()
    final_answer_status = "IN_PROGRESS"

    # -----------------------------------------------------------------------
    # Step 1: Initial Capability Decision
    # -----------------------------------------------------------------------
    try:
        current_decision = decide_initial_capability(client, model_name, user_question)
    except Exception as e:
        err = f"ERROR [Stage: Initial Decision]: Gemini API exception: {e}"
        print(f"\n[ERROR]: {err}")
        print("\n========================================")
        print("END STEP 11")
        print("========================================")
        return err

    if current_decision is None:
        err = "ERROR [Stage: Initial Decision]: Failed to obtain decision from Gemini (Quota 429 / Service 503 / Network error)."
        print(f"\n[ERROR]: {err}")
        print("\n========================================")
        print("END STEP 11")
        print("========================================")
        return err

    # -----------------------------------------------------------------------
    # Step 2: Iterative Execution Loop
    # -----------------------------------------------------------------------
    while iteration <= max_iterations:
        print(f"\n---------------- ITERATION {iteration} ----------------")

        # 2a. Validate Decision
        print("\n[DECISION]")
        print(json.dumps(current_decision, indent=2))

        is_valid, val_msg = validate_selector_decision(current_decision)
        if not is_valid:
            err = f"ERROR [Stage: Decision Validation (Iteration {iteration})]: {val_msg}"
            print(f"\n[VALIDATION ERROR]: {err}")
            break

        # Check duplicate action
        topic_override = current_decision.get("topic") or current_decision.get("topic_override")
        sig = f"{current_decision.get('capability')}:{current_decision.get('recommended_capabilities', [])}:{topic_override or ''}"
        if sig in executed_signatures:
            print(f"\n[DUPLICATE ACTION DETECTED]: Gemini requested already executed capability '{sig}'.")
            print("Halting loop to prevent redundant execution cycles.")
            final_answer_status = "STOPPED_DUPLICATE"
            break
        executed_signatures.add(sig)

        # 2b. Execute Capability via MCP
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                obs = await execute_decision(
                    user_request=user_question,
                    decision=current_decision,
                    topic_override=topic_override,
                )
            exec_trace = _extract_execution_trace(buffer.getvalue())
        except Exception as e:
            err = f"ERROR [Stage: MCP Execution (Iteration {iteration})]: {e}"
            print(f"\n[EXECUTION ERROR]: {err}")
            break

        print("\n[EXECUTION]")
        print(exec_trace)

        # 2c. Store Observation
        observations.append(obs)
        previous_decisions.append(current_decision)

        print("\n[OBSERVATION]")
        if isinstance(obs, dict):
            print(json.dumps(obs, indent=2))
        elif isinstance(obs, str):
            print(obs)
        else:
            print(str(obs))

        # 2d. Evaluate Observations (Gemini Evaluator)
        print("\n[EVALUATION]")

        # If capability was DIRECT, no external data is needed; mark COMPLETE
        if current_decision.get("capability") == "DIRECT":
            eval_result = {
                "status": "COMPLETE",
                "reason": "Conceptual question answered directly without external MCP tools or resources.",
                "next_capability": None,
            }
        else:
            # Polite pause to avoid Free Tier burst rate limits
            await asyncio.sleep(1.0)
            eval_result = evaluate_observations(
                client=client,
                model_name=model_name,
                user_question=user_question,
                previous_decisions=previous_decisions,
                observations=observations,
            )

        if eval_result is None:
            err = f"ERROR [Stage: Gemini Evaluation (Iteration {iteration})]: Failed to obtain evaluation from Gemini."
            print(err)
            final_answer_status = "EVALUATION_FAILED"
            break

        print(json.dumps(eval_result, indent=2))

        # Validate evaluation schema
        is_valid_eval, eval_val_msg = validate_evaluation(eval_result)
        if not is_valid_eval:
            err = f"ERROR [Stage: Evaluation Validation (Iteration {iteration})]: {eval_val_msg}"
            print(f"\n[EVALUATION VALIDATION ERROR]: {err}")
            final_answer_status = "EVALUATION_INVALID"
            break

        # Check status
        if eval_result.get("status") == "COMPLETE":
            final_answer_status = "COMPLETE"
            break

        # If NEEDS_MORE, prepare for next iteration
        next_cap = eval_result.get("next_capability", {})
        current_decision = {
            "capability": next_cap.get("capability"),
            "reason": eval_result.get("reason", "Follow-up required by evaluator"),
            "recommended_capabilities": [next_cap.get("name")] if next_cap.get("name") else [],
            "topic_override": next_cap.get("topic"),
        }

        iteration += 1
        if iteration > max_iterations:
            print(f"\n[NOTICE]: Safety iteration limit (MAX_ITERATIONS = {max_iterations}) reached.")
            final_answer_status = "LIMIT_REACHED"
            break

    # -----------------------------------------------------------------------
    # Step 3: Final Answer Generation (Step 10.3)
    # -----------------------------------------------------------------------
    print("\n================ FINAL ANSWER ================")

    if not observations:
        err = "ERROR: No observations available to synthesize final answer."
        print(err)
        print("\n========================================")
        print("END STEP 11")
        print("========================================")
        return err

    # Synthesize across all gathered observations
    if len(observations) == 1:
        composite_obs = observations[0]
        composite_decision = previous_decisions[0]
    else:
        composite_obs = {
            f"observation_{idx}": obs
            for idx, obs in enumerate(observations, start=1)
        }
        composite_decision = {
            "capability": "MULTI",
            "reason": f"Synthesized from {len(observations)} iterative observation(s).",
            "recommended_capabilities": [
                cap
                for d in previous_decisions
                for cap in d.get("recommended_capabilities", [])
            ],
        }

    try:
        final_answer = generate_final_answer(
            user_question=user_question,
            decision=composite_decision,
            observation=composite_obs,
        )
    except Exception as e:
        final_answer = f"ERROR [Stage: Final Answer Generation]: Gemini API exception: {e}"

    if not final_answer or final_answer.startswith("Status: Gemini") or final_answer.startswith("CONFIGURATION ERROR"):
        final_answer = f"ERROR [Stage: Final Answer Generation]: {final_answer}"

    print(final_answer)

    print("\n========================================")
    print("END STEP 11")
    print("========================================")

    return final_answer


def run_iterative_agent_sync(
    user_question: str,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """Synchronous convenience wrapper for run_iterative_agent."""
    return asyncio.run(run_iterative_agent(user_question, max_iterations))


async def run_tests() -> None:
    """Runs the 3 required controlled iterative integration tests sequentially."""
    test_cases = [
        # TEST 1: Multi-step request
        {
            "id": "TEST 1",
            "name": "MULTI-STEP INFORMATION REQUEST",
            "question": "Who created Java and give me the Spring Boot learning roadmap.",
            "expected_behavior": "Multiple iterations (TOOL -> search_wikipedia, then RESOURCE -> roadmap)",
        },
        # TEST 2: Single-step request
        {
            "id": "TEST 2",
            "name": "SINGLE-STEP REQUEST",
            "question": "Who created Java?",
            "expected_behavior": "Single iteration (TOOL -> search_wikipedia, then COMPLETE)",
        },
        # TEST 3: Direct request
        {
            "id": "TEST 3",
            "name": "DIRECT REQUEST",
            "question": "What is a variable?",
            "expected_behavior": "Single iteration (DIRECT, then COMPLETE)",
        },
    ]

    results = []

    for idx, tc in enumerate(test_cases, start=1):
        if idx > 1:
            # Respect Gemini Free Tier rate limits between test runs
            print("\nWaiting 4 seconds between tests to respect Gemini Free Tier quota...")
            await asyncio.sleep(4)

        print(f"\n##################################################")
        print(f"### RUNNING {tc['id']} — {tc['name']}")
        print(f"### Expected: {tc['expected_behavior']}")
        print(f"##################################################\n")

        answer = await run_iterative_agent(tc["question"])

        if answer.startswith("ERROR"):
            results.append((tc["id"], "FAIL", answer))
        else:
            results.append((tc["id"], "PASS", "Answer generated successfully"))

    print("\n========================================")
    print("STEP 11 TEST SUMMARY")
    print("========================================")
    for test_id, status, detail in results:
        print(f"{test_id}: {status} - {detail}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        custom_question = " ".join(sys.argv[1:])
        asyncio.run(run_iterative_agent(custom_question))
    else:
        asyncio.run(run_tests())

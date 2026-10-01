"""src/agent_mcp/decision_executor.py

STEP 10.2: Decision → Execution
--------------------------------
Bridges the validated Step 10.1 decision with actual MCP execution:
Accepts a validated classification decision (TOOL, RESOURCE, PROMPT, MULTI, or DIRECT)
and orchestrates the corresponding call through the MCP Client to the MCP Server.

ARCHITECTURAL PRINCIPLE:
------------------------
1. Gemini makes decisions (Step 10.1).
2. Python Agent validates the decision structure and whitelists capabilities.
3. Python Agent (NOT Gemini) communicates with the MCP Client.
4. MCP Client executes over stdio transport to the MCP Server.
5. MCP Server executes the requested Tool, Resource, or Prompt.
6. The Observation is captured and returned to the Python Agent.

       User Request
            ↓
    Step 10.1 Gemini Decision
            ↓
    Python Agent Validates Decision
            ↓
        MCP Client (Stdio / JSON-RPC)
            ↓
        MCP Server
        ├── Tool:     get_learning_topics, search_wikipedia
        ├── Resource: learning://spring-boot/roadmap
        └── Prompt:   spring_boot_teacher
            ↓
        Observation (Returned Result)
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ensure clean UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Whitelist constants
VALID_CATEGORIES = {"TOOL", "RESOURCE", "PROMPT", "MULTI", "DIRECT"}

ALLOWED_CAPABILITIES = {
    "get_learning_topics",
    "search_wikipedia",
    "get_student_profile",
    "get_student_skills",
    "learning://spring-boot/roadmap",
    "spring_boot_teacher",
}


# ---------------------------------------------------------------------------
# 1. Validation Layer
# ---------------------------------------------------------------------------
def validate_decision(decision: dict[str, Any]) -> tuple[bool, str]:
    """Strictly validates the decision object before any execution begins."""
    if not isinstance(decision, dict):
        return False, "Decision is not a dictionary."

    capability = decision.get("capability")
    if capability not in VALID_CATEGORIES:
        return False, f"Unknown capability '{capability}'. Allowed: {sorted(VALID_CATEGORIES)}."

    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False, "Field 'reason' must be a non-empty string."

    recommended = decision.get("recommended_capabilities")
    if not isinstance(recommended, list):
        return False, "Field 'recommended_capabilities' must be a list."

    for item in recommended:
        if item not in ALLOWED_CAPABILITIES:
            return False, f"Unauthorized capability '{item}'. Allowed: {sorted(ALLOWED_CAPABILITIES)}."

    if capability == "DIRECT" and len(recommended) != 0:
        return False, f"DIRECT category requires empty recommended_capabilities, found: {recommended}."

    if capability in {"TOOL", "RESOURCE", "PROMPT"} and len(recommended) != 1:
        return False, f"Category '{capability}' expects exactly 1 recommended capability, found: {recommended}."

    if capability == "MULTI":
        if len(recommended) < 2:
            return False, f"MULTI category requires 2 or more recommended capabilities, found: {recommended}."

    return True, "PASS"


# ---------------------------------------------------------------------------
# 2. Topic Extraction Helper
# ---------------------------------------------------------------------------
def extract_topic(user_request: str, default: str = "Spring Boot") -> str:
    """Extracts the subject keyword from the user request."""
    q = user_request.lower()
    if "dependency injection" in q:
        return "Dependency Injection"
    elif "spring security" in q:
        return "Spring Boot Security"
    elif "spring boot" in q:
        return "Spring Boot"
    elif "java" in q:
        return "Java"
    elif "docker" in q:
        return "Docker"
    elif "python" in q:
        return "Python"
    return default


# ---------------------------------------------------------------------------
# 3. Decision Executor
# ---------------------------------------------------------------------------
async def execute_decision(
    user_request: str,
    decision: dict[str, Any],
    topic_override: str | None = None,
) -> dict[str, Any]:
    """Validates the decision and executes the corresponding MCP capability."""
    print("==================================================")
    print("STEP 10.2 — DECISION → EXECUTION")
    print("==================================================")
    print(f"\nUser Request:\n\"{user_request}\"")
    print(f"\nDecision:\n{json.dumps(decision, indent=2)}")

    # Step A: Validate decision
    is_valid, validation_msg = validate_decision(decision)
    print(f"\nValidation:\n{validation_msg}")
    if not is_valid:
        raise ValueError(f"Decision validation failed: {validation_msg}")

    capability = decision["capability"]
    recommended = decision["recommended_capabilities"]
    topic = topic_override or extract_topic(user_request)

    # -----------------------------------------------------------------------
    # Case 1: DIRECT (No MCP Execution)
    # -----------------------------------------------------------------------
    if capability == "DIRECT":
        print("\nExecution:")
        print("Capability: DIRECT")
        print("Selected: None")
        print("\nPython Agent")
        print("    ↓")
        print("[No MCP execution required — request handled directly]")

        observation = {
            "status": "DIRECT",
            "message": "No MCP execution required. The question can be answered directly by the LLM without external tools or data.",
        }
        print(f"\nObservation:\n{json.dumps(observation, indent=2)}")
        return observation

    # -----------------------------------------------------------------------
    # Cases 2-5: TOOL, RESOURCE, PROMPT, MULTI (Execute via MCP Client)
    # -----------------------------------------------------------------------
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["src/mcp_server/server.py"],
        env=dict(os.environ),
    )

    observation: dict[str, Any] = {}

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # --- Case 2: TOOL Execution ---
            if capability == "TOOL":
                tool_name = recommended[0]
                if tool_name in {"get_student_profile", "get_student_skills"}:
                    tool_args: dict[str, Any] = {}
                else:
                    tool_args = {"topic": topic}

                print("\nExecution:")
                print(f"Capability: TOOL")
                print(f"Selected: {tool_name}")
                print(f"Arguments: {json.dumps(tool_args)}")
                print("\nPython Agent")
                print("    ↓")
                print("MCP Client")
                print("    ↓")
                print("MCP Server")
                print("    ↓")
                print(f"{tool_name}")

                tool_result = await session.call_tool(tool_name, tool_args)
                for c in tool_result.content:
                    if c.type == "text":
                        try:
                            observation = json.loads(c.text)
                        except json.JSONDecodeError:
                            observation = {"raw": c.text}

            # --- Case 3: RESOURCE Execution ---
            elif capability == "RESOURCE":
                resource_uri = recommended[0]
                print("\nExecution:")
                print(f"Capability: RESOURCE")
                print(f"Selected: {resource_uri}")
                print("\nPython Agent")
                print("    ↓")
                print("MCP Client")
                print("    ↓")
                print("MCP Server")
                print("    ↓")
                print(f"{resource_uri}")

                res_result = await session.read_resource(resource_uri)
                if res_result.contents:
                    observation = {
                        "uri": resource_uri,
                        "content": res_result.contents[0].text,
                    }
                else:
                    observation = {"uri": resource_uri, "content": "Empty resource"}

            # --- Case 4: PROMPT Execution ---
            elif capability == "PROMPT":
                prompt_name = recommended[0]
                print("\nExecution:")
                print(f"Capability: PROMPT")
                print(f"Selected: {prompt_name}")
                print(f"Arguments: {{\"topic\": \"{topic}\"}}")
                print("\nPython Agent")
                print("    ↓")
                print("MCP Client")
                print("    ↓")
                print("MCP Server")
                print("    ↓")
                print(f"{prompt_name}")

                prompt_result = await session.get_prompt(prompt_name, {"topic": topic})
                prompt_text = ""
                if prompt_result.messages and hasattr(prompt_result.messages[0].content, "text"):
                    prompt_text = prompt_result.messages[0].content.text
                else:
                    prompt_text = str(prompt_result.messages)

                observation = {
                    "prompt": prompt_name,
                    "topic": topic,
                    "instructions": prompt_text,
                }

            # --- Case 5: MULTI Execution ---
            elif capability == "MULTI":
                print("\nExecution:")
                print(f"Capability: MULTI")
                print(f"Selected: {recommended}")
                print("\nPython Agent")
                print("    ↓")
                print("MCP Client")
                print("    ↓")
                print("MCP Server")

                multi_observation: dict[str, Any] = {}

                for cap_id in recommended:
                    # TOOL capability execution
                    if cap_id in {"get_student_profile", "get_student_skills", "get_learning_topics", "search_wikipedia"}:
                        if cap_id in {"get_student_profile", "get_student_skills"}:
                            t_args: dict[str, Any] = {}
                        else:
                            t_args = {"topic": topic}

                        print(f"    ├── Calling Tool: {cap_id} with {json.dumps(t_args)}")
                        t_result = await session.call_tool(cap_id, t_args)
                        t_obs: Any = None
                        for c in t_result.content:
                            if c.type == "text":
                                try:
                                    t_obs = json.loads(c.text)
                                except json.JSONDecodeError:
                                    t_obs = {"raw": c.text}
                        multi_observation[cap_id] = t_obs

                    # RESOURCE capability execution
                    elif cap_id.startswith("learning://"):
                        print(f"    ├── Reading Resource: {cap_id}")
                        resource_res = await session.read_resource(cap_id)
                        r_text = resource_res.contents[0].text if resource_res.contents else ""
                        multi_observation["resource"] = {
                            "uri": cap_id,
                            "content": r_text,
                        }

                    # PROMPT capability execution
                    elif cap_id == "spring_boot_teacher":
                        print(f"    ├── Retrieving Prompt: {cap_id}")
                        prompt_res = await session.get_prompt(cap_id, {"topic": topic})
                        p_text = prompt_res.messages[0].content.text if prompt_res.messages else ""
                        multi_observation["prompt"] = {
                            "name": cap_id,
                            "instructions": p_text,
                        }

                observation = multi_observation

    # Print captured observation
    print(f"\nObservation:")
    if "content" in observation and isinstance(observation["content"], str) and "\n" in observation["content"]:
        # Pretty print text resources
        print(f"URI: {observation.get('uri')}")
        print(f"Content:\n{observation['content']}")
    elif "instructions" in observation and isinstance(observation["instructions"], str):
        # Pretty print prompt text
        print(f"Prompt: {observation.get('prompt')}")
        print(f"Instructions:\n{observation['instructions']}")
    elif "prompt" in observation and "resource" in observation:
        # Multi execution
        print(json.dumps({
            "prompt": observation["prompt"]["name"],
            "prompt_instructions_preview": observation["prompt"]["instructions"][:120] + "...",
            "resource": observation["resource"]["uri"],
            "resource_content_preview": observation["resource"]["content"][:120] + "...",
        }, indent=2))
    else:
        print(json.dumps(observation, indent=2))

    return observation


# ---------------------------------------------------------------------------
# 4. Controlled Test Suite (5 Required Tests)
# ---------------------------------------------------------------------------
async def run_controlled_tests() -> None:
    """Executes the 5 required controlled tests with predefined Step 10.1 decisions."""
    test_cases = [
        # TEST 1
        {
            "id": "TEST 1",
            "request": "Who created Java?",
            "decision": {
                "capability": "TOOL",
                "reason": "Requires factual and historical information about Java from Wikipedia.",
                "recommended_capabilities": ["search_wikipedia"],
            },
            "topic": "Java",
            "expected_capability": "TOOL",
        },
        # TEST 2
        {
            "id": "TEST 2",
            "request": "Show me the Spring Boot learning roadmap.",
            "decision": {
                "capability": "RESOURCE",
                "reason": "Requires existing authoritative Spring Boot roadmap document.",
                "recommended_capabilities": ["learning://spring-boot/roadmap"],
            },
            "topic": "Spring Boot",
            "expected_capability": "RESOURCE",
        },
        # TEST 3
        {
            "id": "TEST 3",
            "request": "Teach me Dependency Injection like a beginner.",
            "decision": {
                "capability": "PROMPT",
                "reason": "Requires beginner-friendly pedagogical teaching instructions.",
                "recommended_capabilities": ["spring_boot_teacher"],
            },
            "topic": "Dependency Injection",
            "expected_capability": "PROMPT",
        },
        # TEST 4
        {
            "id": "TEST 4",
            "request": "Teach me Spring Security and explain where it fits in the Spring Boot roadmap.",
            "decision": {
                "capability": "MULTI",
                "reason": "Requires teaching instructions and roadmap context.",
                "recommended_capabilities": [
                    "spring_boot_teacher",
                    "learning://spring-boot/roadmap",
                ],
            },
            "topic": "Spring Boot Security",
            "expected_capability": "MULTI",
        },
        # TEST 5
        {
            "id": "TEST 5",
            "request": "What is a variable?",
            "decision": {
                "capability": "DIRECT",
                "reason": "Basic conceptual question that does not require external tools or data.",
                "recommended_capabilities": [],
            },
            "topic": "General",
            "expected_capability": "DIRECT",
        },
    ]

    results = []

    for tc in test_cases:
        print(f"\n##################################################")
        print(f"### {tc['id']}")
        print(f"##################################################")
        try:
            obs = await execute_decision(
                user_request=tc["request"],
                decision=tc["decision"],
                topic_override=tc["topic"],
            )
            # Verify observation was generated
            if obs:
                results.append((tc["id"], "PASS", tc["expected_capability"]))
            else:
                results.append((tc["id"], "FAIL (Empty Observation)", tc["expected_capability"]))
        except Exception as e:
            results.append((tc["id"], f"FAIL ({e})", tc["expected_capability"]))

    print("\n==================================================")
    print("STEP 10.2 TEST SUMMARY")
    print("==================================================")
    for test_id, status, cap in results:
        print(f"{test_id} [{cap}]: {status}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run_controlled_tests())

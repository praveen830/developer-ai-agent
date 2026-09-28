"""src/agent_mcp/agent_loop.py

STEP 6: The Agent Loop Architecture
-----------------------------------

WHAT IS AN AGENT LOOP?
An Agent Loop is the fundamental operational cycle that turns a static model
or script into an autonomous agent. Instead of a single one-and-done execution,
the agent executes a cyclical feedback loop:
    Perceive / Understand -> Decide Action -> Execute Action -> Observe Result -> Evaluate

WHAT IS AN ITERATION?
An iteration is a single pass through the cycle. During each iteration, the agent
takes one concrete step toward solving the user's goal. Complex multi-step tasks
(e.g., researching, planning, executing code, testing, fixing errors) require
multiple iterations.

WHAT IS AN ACTION?
An Action is a discrete decision taken by the agent to affect its environment or
gather information. In our architecture, an Action is a tool call (such as invoking
the MCP tool `get_learning_topics` with specific arguments).

WHAT IS AN OBSERVATION?
An Observation is the raw sensory input or structured data returned back to the
agent after an Action executes. Here, the Observation is the JSON result produced
by the MCP Server tool.

WHAT DOES EVALUATION MEAN?
Evaluation is the reflection stage where the agent verifies whether the Observation
actually answered the user's request or made progress. It assesses:
- Did the tool succeed or fail?
- Is the output valid, complete, and relevant?
- Has the user's overarching goal been satisfied?

WHY DOES `task_complete` EXIST?
`task_complete` is the boolean termination flag. It signals that the evaluation
criteria have been met and no further actions are necessary, allowing the loop
to terminate cleanly and yield the final response.

WHY IS `MAX_ITERATIONS` NECESSARY?
Without an upper boundary, an agent can get trapped in an infinite loop due to
repeating errors, failed tool invocations, or unsolvable tasks. `MAX_ITERATIONS`
acts as a safety circuit-breaker to guarantee termination.

WHY MIGHT AN AGENT EXECUTE ANOTHER ACTION IN A FUTURE ITERATION?
In real-world multi-step tasks, one action's observation informs the next action.
For example, an agent might:
  Iteration 1: Call `get_learning_topics` to fetch curriculum modules.
  Iteration 2: Call `search_documentation` for a specific difficult module.
  Iteration 3: Call `generate_quiz` to test understanding.
If an action fails, the evaluation marks `task_complete = False`, prompting the agent
to retry with adjusted arguments in the next iteration.

WHY IS THE MCP CLIENT RESPONSIBLE FOR COMMUNICATING WITH THE MCP SERVER?
The agent remains agnostic of the server's runtime environment (process, container,
or remote network). The MCP Client abstracts JSON-RPC protocol serialization,
stdio transport pipes, and handshake lifecycles, maintaining a clean boundary.
"""

import asyncio
import json
import os
import re
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


# Safety guard: maximum cycles permitted before forceful termination
MAX_ITERATIONS = 5


# ---------------------------------------------------------------------------
# Natural Language Understanding & Parameter Extraction
# ---------------------------------------------------------------------------
def understand_user_request(query: str) -> tuple[bool, str]:
    """Analyzes the user request to determine intent and extract topic.

    Returns:
        (is_roadmap_request, topic_name)
    """
    roadmap_triggers = [
        "roadmap",
        "learning path",
        "topics to learn",
        "study plan",
        "what to learn",
        "how to learn",
        "curriculum",
    ]
    query_lower = query.lower()
    is_roadmap = any(trigger in query_lower for trigger in roadmap_triggers)

    # Extract technical topic
    if "spring boot" in query_lower or "spring" in query_lower:
        topic = "Spring Boot"
    elif "docker" in query_lower:
        topic = "Docker"
    elif "python" in query_lower:
        topic = "Python"
    else:
        # Fallback: strip trigger keywords to isolate subject
        cleaned = re.sub(
            r"(give me|a|learning|roadmap|path|study plan|topics to learn|for|about)",
            "",
            query,
            flags=re.IGNORECASE,
        ).strip()
        topic = cleaned.title() if cleaned else "General Programming"

    return is_roadmap, topic


# ---------------------------------------------------------------------------
# MCP Client Bridge: Communicates with the MCP Server over stdio
# ---------------------------------------------------------------------------
async def execute_mcp_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Connects to the MCP server via the MCP Client, calls the tool, and returns result."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["src/mcp_server/server.py"],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)

            # Parse returned MCP content
            for content in result.content:
                if content.type == "text":
                    try:
                        return json.loads(content.text)
                    except json.JSONDecodeError:
                        return {"raw": content.text}

            if result.structured_content:
                return result.structured_content

            return {}


# ---------------------------------------------------------------------------
# Evaluation Logic: Verifies if Observation satisfies the goal
# ---------------------------------------------------------------------------
def evaluate_observation(observation: dict[str, Any]) -> tuple[bool, str]:
    """Evaluates whether the tool output contains valid, non-empty learning topics.

    Returns:
        (is_valid, evaluation_reason)
    """
    if not isinstance(observation, dict):
        return False, "Invalid result format: Observation is not a structured dictionary."

    if "error" in observation:
        return False, f"Tool reported an error: {observation['error']}"

    topics = observation.get("topics")
    if not isinstance(topics, list) or len(topics) == 0:
        return False, "Observation does not contain a valid non-empty 'topics' list."

    return True, "Valid learning topics received."


# ---------------------------------------------------------------------------
# Main Agent Loop Orchestrator
# ---------------------------------------------------------------------------
async def run_agent(user_request: str) -> None:
    """Executes the autonomous agent loop through iterative cycles."""
    print("==================================================")
    print("AGENT LOOP")
    print("==================================================")
    print(f"\nUSER REQUEST:\n{user_request}")

    iteration = 0
    task_complete = False
    observation: dict[str, Any] = {}
    failure_reason = ""

    # -----------------------------------------------------------------------
    # THE AGENT LOOP
    # The loop keeps cycling until the task is evaluated as complete
    # or the safety iteration limit is reached.
    # -----------------------------------------------------------------------
    while not task_complete and iteration < MAX_ITERATIONS:
        iteration += 1

        print(f"\n--------------------------------------------------")
        print(f"ITERATION {iteration}")
        print(f"--------------------------------------------------")

        # 1. UNDERSTAND REQUEST
        is_roadmap, topic = understand_user_request(user_request)
        if is_roadmap:
            print("\nUNDERSTAND:\nLearning roadmap requested.")
        else:
            print("\nUNDERSTAND:\nGeneral technical query requested.")

        # 2. DECIDE ACTION
        # For a roadmap request, the agent chooses the MCP tool 'get_learning_topics'
        action = "get_learning_topics"
        action_args = {"topic": topic}

        print(f"\nACTION:\n{action}")
        print(f"\nARGUMENTS:\n{json.dumps(action_args, indent=2)}")

        # 3. EXECUTE ACTION
        # The agent delegates execution to the MCP Client
        print("\nEXECUTING MCP TOOL...")
        try:
            raw_result = await execute_mcp_tool(action, action_args)
            observation = raw_result
        except Exception as e:
            observation = {"error": f"MCP connection/execution failure: {str(e)}"}

        # 4. OBSERVE RESULT
        print(f"\nOBSERVATION:\n{json.dumps(observation, indent=2)}")

        # 5. EVALUATE RESULT
        # Assess if the observation satisfies the task
        is_valid, eval_message = evaluate_observation(observation)
        print(f"\nEVALUATION:\n{eval_message}")

        if is_valid:
            task_complete = True
            print("\nTASK COMPLETE:\ntrue")
        else:
            task_complete = False
            failure_reason = eval_message
            print("\nTASK COMPLETE:\nfalse")
            # If invalid, a real agent might adapt arguments or try another tool on iteration + 1

    # -----------------------------------------------------------------------
    # FINAL RESPONSE
    # Formats the final answer after the loop exits cleanly
    # -----------------------------------------------------------------------
    print("\n--------------------------------------------------")
    print("FINAL RESPONSE")
    print("--------------------------------------------------\n")

    if task_complete:
        topic_name = observation.get("topic", "Technical")
        topics_list = observation.get("topics", [])

        print(f"{topic_name} learning roadmap:\n")
        for i, item in enumerate(topics_list, start=1):
            print(f"{i}. {item}")
    else:
        if iteration >= MAX_ITERATIONS:
            print(f"Agent loop reached MAX_ITERATIONS ({MAX_ITERATIONS}) without completing task.")
        print(f"Could not complete roadmap generation: {failure_reason}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    request = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Give me a Spring Boot learning roadmap"
    )
    asyncio.run(run_agent(request))

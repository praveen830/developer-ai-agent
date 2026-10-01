"""src/agent_mcp/advanced_agent_loop.py

STEP 8: Advanced Gemini-Powered Agent Loop + MCP Tools
------------------------------------------------------

AGENTIC PATTERN: Action → Observation → Evaluation → Next Action
----------------------------------------------------------------
This module implements a true agent loop governed by the iterative cycle:

1. ACTION:
   Gemini decides what step to take next based on the user request and any
   accumulated context. If it needs external facts or roadmaps, it issues a
   tool call action (e.g. calling `search_wikipedia` or `get_learning_topics`).

2. OBSERVATION:
   The Python Agent validates the tool call, executes it safely via the MCP Client
   over stdio to the MCP Server, and collects the returned data. This returned data
   is the "Observation".

3. EVALUATION:
   The Observation is packaged into the conversation history and sent back to Gemini.
   Gemini evaluates whether the new information is sufficient to solve the user's
   request or whether additional information is needed.

4. NEXT ACTION (OR TERMINATION):
   - If more information is needed, Gemini issues another tool call (Next Action).
   - If the task is complete, Gemini produces the final natural-language response.
   - If the safety limit (MAX_ITERATIONS) is reached, Python halts execution.

ARCHITECTURAL PRINCIPLE:
------------------------
Gemini NEVER directly connects to the MCP Server or external network sockets.
The Python Agent acts as the orchestrator and security boundary:

       User Request
            ↓
       Python Agent (Loop Controller)
            ↓
          Gemini (Reasoning Engine)
            ↓
    [Gemini Decision]
      ├─► Tool Call requested?
      │     ↓
      │   Python Agent (Validates against ALLOWED_TOOLS whitelist)
      │     ↓
      │   MCP Client (JSON-RPC over stdio)
      │     ↓
      │   MCP Server (Executes tool logic)
      │     ↓
      │   MCP Tool Result (Observation)
      │     ↓
      │   Append to conversation history & loop to next iteration
      │
      └─► No tool requested? (Task Complete)
            ↓
          Final Natural-Language Answer
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ensure clean UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress verbose informational logs from the Google GenAI SDK
logging.getLogger("google_genai").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Constants & Safety Controls
# ---------------------------------------------------------------------------
# Maximum iterations permitted before forceful termination to prevent infinite loops
MAX_ITERATIONS = 5

# Strict whitelist: only these MCP tools are permitted to execute
ALLOWED_TOOLS = [
    "get_learning_topics",
    "search_wikipedia",
]


# ---------------------------------------------------------------------------
# 1. API Key Retrieval (Safe: reads environment without exposing credentials)
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    """Read GEMINI_API_KEY from environment or Windows User registry fallback."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    key = os.environ.get("GEMINI_API_KEY")
    if not key and sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as reg_key:
                key, _ = winreg.QueryValueEx(reg_key, "GEMINI_API_KEY")
        except Exception:
            pass

        if not key:
            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                ) as reg_key:
                    key, _ = winreg.QueryValueEx(reg_key, "GEMINI_API_KEY")
            except Exception:
                pass
    return key


# ---------------------------------------------------------------------------
# 2. Tool Declarations & Schemas for Gemini
# ---------------------------------------------------------------------------
# We declare both MCP tools to Gemini so it understands what capabilities
# are available and the arguments they require.
tools_declaration = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_learning_topics",
            description="Get a structured list of learning topics for a requested subject or technology.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "topic": types.Schema(
                        type="STRING",
                        description="The requested subject or technology.",
                    ),
                },
                required=["topic"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_wikipedia",
            description="Search Wikipedia for factual/reference information about a requested topic.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "topic": types.Schema(
                        type="STRING",
                        description="The requested topic.",
                    ),
                },
                required=["topic"],
            ),
        ),
    ]
)


# ---------------------------------------------------------------------------
# 3. MCP Client Bridge: Communicates with MCP Server over stdio
# ---------------------------------------------------------------------------
async def execute_mcp_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Connects to the MCP server via the MCP Client over stdio transport.
    
    Spawns 'src/mcp_server/server.py' as a subprocess, initializes the JSON-RPC
    session, calls the selected tool, and returns the parsed result.
    Gemini does NOT have direct access to this connection.
    """
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
# 4. Advanced Iterative Agent Loop
# ---------------------------------------------------------------------------
async def run_advanced_agent_loop(user_query: str) -> None:
    """Executes the advanced iterative agent loop:
    Action -> Observation -> Evaluation -> Next Action / Completion
    """
    # Verify API key presence
    api_key = get_api_key()
    if not api_key:
        print(
            "CONFIGURATION ERROR: GEMINI_API_KEY environment variable is not set.\n"
            "Please configure GEMINI_API_KEY in your environment or .env file.",
            file=sys.stderr,
        )
        return

    print("==================================================")
    print("ADVANCED GEMINI AGENT LOOP")
    print("==================================================")
    print(f"\nUSER REQUEST:\n{user_query}")
    sys.stdout.flush()

    # Initialize Gemini client & model
    client = genai.Client(api_key=api_key)
    # Default to gemini-flash-latest for reliable Free Tier operation
    model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

    # Conversation history: accumulates user prompt, assistant tool calls, and tool results
    contents: list[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )
    ]

    iteration = 0
    task_complete = False

    # -----------------------------------------------------------------------
    # THE ITERATIVE LOOP
    # Cycles through Action -> Observation -> Evaluation until Gemini
    # produces a final answer or MAX_ITERATIONS is reached.
    # -----------------------------------------------------------------------
    while not task_complete and iteration < MAX_ITERATIONS:
        iteration += 1

        if iteration > 1:
            # Respect Free Tier burst rate limits between agent loop cycles
            await asyncio.sleep(2)

        print(f"\n--------------------------------------------------")
        print(f"ITERATION {iteration}")
        print(f"--------------------------------------------------")

        # Step A: Query Gemini with current conversation history and available tools
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[tools_declaration],
                    temperature=0.0,  # Deterministic decisions
                ),
            )
        except ClientError as e:
            if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
                print(
                    f"\nGEMINI API ERROR (429 Quota/Rate Limit Exhausted):\n"
                    f"Gemini Free Tier quota reached. Please wait before retrying.\n"
                    f"Details: {e}",
                    file=sys.stderr,
                )
            else:
                print(f"\nGEMINI API ERROR ({e.code}):\n{e}", file=sys.stderr)
            return
        except ServerError as e:
            if e.code == 503 or "UNAVAILABLE" in str(e):
                print(
                    f"\nGEMINI API ERROR (503 Service Unavailable):\n"
                    f"Gemini service is temporarily unavailable. Please try again shortly.\n"
                    f"Details: {e}",
                    file=sys.stderr,
                )
            else:
                print(f"\nGEMINI API SERVER ERROR ({e.code}):\n{e}", file=sys.stderr)
            return
        except Exception as e:
            print(f"\nGEMINI API UNEXPECTED ERROR:\n{e}", file=sys.stderr)
            return

        print("\nGEMINI DECISION")

        # Step B: Check if Gemini decided a tool call is needed
        if not response.function_calls:
            # Case 1: No tool requested -> Gemini has evaluated that the task is complete
            task_complete = True
            print("\nGEMINI DID NOT REQUEST AN MCP TOOL (TASK COMPLETE)")
            print("\n==================================================")
            print("FINAL RESPONSE")
            print("==================================================")
            final_answer = response.text.strip() if response.text else "No response generated."
            print(f"\n{final_answer}")
            break

        # Case 2: Gemini requested a tool call (Action)
        call = response.function_calls[0]
        tool_name = call.name
        tool_args = call.args or {}

        print(f"\nSELECTED TOOL:\n{tool_name}")
        print(f"\nARGUMENTS:\n{json.dumps(tool_args, indent=2)}")

        # Step C: Validate tool name against whitelist
        if tool_name not in ALLOWED_TOOLS:
            print(
                f"\nError: Unknown tool '{tool_name}' requested by Gemini.\n"
                f"Allowed tools: {ALLOWED_TOOLS}. Execution aborted.",
                file=sys.stderr,
            )
            return

        # Validate arguments structure
        if (
            not isinstance(tool_args, dict)
            or "topic" not in tool_args
            or not isinstance(tool_args["topic"], str)
            or not tool_args["topic"].strip()
        ):
            print(
                f"\nError: Malformed arguments received from Gemini for tool '{tool_name}': {tool_args}",
                file=sys.stderr,
            )
            return

        # Step D: Execute MCP Tool via MCP Client (Calling MCP Server)
        try:
            tool_result = await execute_mcp_tool(tool_name, tool_args)
        except Exception as e:
            print(f"\nMCP TOOL EXECUTION ERROR:\nFailed to execute tool on MCP server: {e}", file=sys.stderr)
            tool_result = {"error": f"MCP execution failed: {str(e)}"}

        # Step E: Print Observation (MCP Tool Result)
        print("\n==================================================")
        print("MCP TOOL RESULT")
        print("==================================================")
        print(f"\n{json.dumps(tool_result, indent=2)}")

        # Step F: Update Conversation History (Sending tool result back to Gemini)
        # 1. Append the model's function call turn
        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        # 2. Append the function response (Observation)
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_result},
                    )
                ],
            )
        )
        # The loop now proceeds to the next iteration:
        # Gemini will inspect the tool result in the next turn and decide whether
        # to request another tool or finalize its response.

    # -----------------------------------------------------------------------
    # Termination Safety Check
    # -----------------------------------------------------------------------
    if not task_complete and iteration >= MAX_ITERATIONS:
        print("\n==================================================")
        print(f"TERMINATION: MAX ITERATIONS REACHED ({MAX_ITERATIONS})")
        print("==================================================")
        print(
            f"The agent loop reached its maximum iteration safety limit ({MAX_ITERATIONS}) "
            "without completing the task. Stopping to prevent an infinite loop."
        )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is Docker and give me a learning roadmap for it?"
    )
    asyncio.run(run_advanced_agent_loop(query))

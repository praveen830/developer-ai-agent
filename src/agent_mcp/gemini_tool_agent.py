"""src/agent_mcp/gemini_tool_agent.py

STEP 7D: Gemini-Based MCP Tool Selection
----------------------------------------

ARCHITECTURAL PRINCIPLE:
------------------------
Gemini does NOT directly communicate with or execute tools on the MCP server.
The Python Agent acts as the central bridge and orchestrator:

       User
        ↓
   Python Agent
        ↓
      Gemini
        ↓
   Gemini selects an MCP tool (JSON Function Call)
        ↓
   Python Agent validates tool name and arguments
        ↓
     MCP Client (Stdio transport / JSON-RPC protocol)
        ↓
     MCP Server (Hosts registered tools)
        ↓
   Selected MCP Tool (e.g. 'get_learning_topics' or 'search_wikipedia')
        ↓
    Tool Result (Structured data)
        ↓
   Python Agent (Packages tool output into conversation context)
        ↓
      Gemini (Synthesizes grounded final answer)
        ↓
    Final Response

BEGINNER-FRIENDLY CONCEPTUAL OVERVIEW:
--------------------------------------
1. GEMINI TOOL DECLARATIONS:
   We provide Gemini with JSON-compatible function signatures (schemas) describing:
   - Tool 1: `get_learning_topics` (structured learning curriculum/roadmap)
   - Tool 2: `search_wikipedia` (factual encyclopedic summaries)
   These declarations teach Gemini what capabilities exist and what inputs they take.

2. TOOL SCHEMA:
   Each tool defines a `parameters` schema specifying data types and required fields
   (e.g., `topic` as a STRING). Gemini uses this schema to generate structured arguments.

3. GEMINI TOOL SELECTION:
   When given a user prompt, Gemini's reasoning determines if an external tool is
   needed. If needed, it outputs a function call containing the tool name and arguments.

4. PYTHON AGENT ORCHESTRATION:
   The Python Agent intercepts Gemini's decision, validates that the requested tool is
   in the approved whitelist, and confirms the arguments are well-formed.

5. MCP CLIENT CALL:
   The Python Agent invokes the MCP Client over a standard I/O (stdio) transport pipe.
   Gemini never has direct network or subprocess access.

6. MCP SERVER EXECUTION:
   The MCP Server receives the JSON-RPC tool call, runs the corresponding Python function,
   and returns the output back across the stdio pipe to the MCP Client.

7. RETURNING THE TOOL RESULT TO GEMINI:
   The Python Agent wraps the tool output in a `function_response` message and appends
   it to the conversation history, sending it back to Gemini.

8. FINAL GEMINI RESPONSE:
   Gemini reads the conversation history including the tool's returned data and
   generates a clear, natural-language response grounded in the tool's findings.
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

# Ensure clean UTF-8 console output on Windows platforms
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress verbose informational logs from Google GenAI SDK
logging.getLogger("google_genai").setLevel(logging.ERROR)


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
# 2. Gemini Tool Declarations & Schemas
# ---------------------------------------------------------------------------
# GEMINI TOOL DECLARATIONS & TOOL SCHEMA:
# We define function declarations that match the existing MCP tools.
# Gemini uses these declarations to understand when and how to call each tool.
tools_declaration = types.Tool(
    function_declarations=[
        # Tool 1: get_learning_topics
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
        # Tool 2: search_wikipedia
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

# Strict whitelist of allowed tools to prevent unauthorized execution
ALLOWED_TOOLS = ["get_learning_topics", "search_wikipedia"]


# ---------------------------------------------------------------------------
# 3. MCP Client Bridge: Communicates with MCP Server over stdio
# ---------------------------------------------------------------------------
async def execute_mcp_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """MCP CLIENT CALL & MCP SERVER EXECUTION:
    Connects to the existing MCP server via the official MCP Client over stdio transport.
    Spawns 'src/mcp_server/server.py' as a subprocess, initializes the JSON-RPC session,
    calls the selected tool, and returns the parsed result.
    
    Gemini does NOT connect to the MCP Server directly.
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

            # Parse returned MCP content parts
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
# 4. Agent Orchestration: Gemini Tool Selection + MCP Bridge + Final Response
# ---------------------------------------------------------------------------
async def run_gemini_tool_agent(user_query: str) -> None:
    """End-to-end workflow:
    1. Validates API key configuration
    2. Sends user query + tool declarations to Gemini
    3. Evaluates Gemini's tool decision
    4. Validates tool name and arguments in Python
    5. Calls MCP Server through MCP Client
    6. Returns tool result to Gemini
    7. Displays Gemini's final natural-language response
    """
    # -----------------------------------------------------------------------
    # Step A: Validate GEMINI_API_KEY
    # -----------------------------------------------------------------------
    api_key = get_api_key()
    if not api_key:
        print(
            "CONFIGURATION ERROR: GEMINI_API_KEY environment variable is not set.\n"
            "Please configure GEMINI_API_KEY in your environment or .env file.",
            file=sys.stderr,
        )
        return

    print("==================================================")
    print("USER REQUEST")
    print("==================================================")
    print(f"\n{user_query}")
    sys.stdout.flush()

    # Initialize Google GenAI client
    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

    # -----------------------------------------------------------------------
    # Step B: Prepare Conversation with User Prompt
    # -----------------------------------------------------------------------
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )
    ]

    # -----------------------------------------------------------------------
    # Step C: Send User Request + Tool Declarations to Gemini
    # -----------------------------------------------------------------------
    # GEMINI TOOL SELECTION:
    # Gemini analyzes the user query and the tool declarations.
    # It decides whether to answer directly or request a tool invocation.
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[tools_declaration],
                temperature=0.0,  # Deterministic tool decision
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

    # -----------------------------------------------------------------------
    # Step D: Inspect Gemini's Tool Decision
    # -----------------------------------------------------------------------
    print("\n==================================================")
    print("GEMINI TOOL DECISION")
    print("==================================================")

    # Case A: Gemini did not request any tool -> answers directly
    if not response.function_calls:
        print("\nGEMINI DID NOT REQUEST AN MCP TOOL")
        print("\n==================================================")
        print("GEMINI FINAL RESPONSE")
        print("==================================================")
        final_answer = response.text.strip() if response.text else "No response generated."
        print(f"\n{final_answer}")
        return

    # Case B: Gemini requested a tool call
    call = response.function_calls[0]
    tool_name = call.name
    tool_args = call.args or {}

    print(f"\nSELECTED TOOL:\n{tool_name}")
    print(f"\nARGUMENTS:\n{json.dumps(tool_args, indent=2)}")

    # -----------------------------------------------------------------------
    # Step E: Python Agent Orchestration & Validation
    # -----------------------------------------------------------------------
    # Validate tool name against whitelist
    if tool_name not in ALLOWED_TOOLS:
        print(
            f"\nError: Unknown tool '{tool_name}' selected by Gemini. "
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

    # -----------------------------------------------------------------------
    # Step F: Execute Tool via MCP Client (Calling MCP Server)
    # -----------------------------------------------------------------------
    try:
        tool_result = await execute_mcp_tool(tool_name, tool_args)
    except Exception as e:
        print(f"\nMCP TOOL EXECUTION ERROR:\nFailed to execute tool on MCP server: {e}", file=sys.stderr)
        return

    # Print MCP Tool Result
    print("\n==================================================")
    print("MCP TOOL RESULT")
    print("==================================================")
    print(f"\n{json.dumps(tool_result, indent=2)}")

    # -----------------------------------------------------------------------
    # Step G: Return Tool Result to Gemini
    # -----------------------------------------------------------------------
    # Append Gemini's previous turn (assistant function call)
    if response.candidates and response.candidates[0].content:
        contents.append(response.candidates[0].content)

    # Append the tool's execution result as a function response
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

    # -----------------------------------------------------------------------
    # Step H: Final Gemini Synthesis
    # -----------------------------------------------------------------------
    # FINAL GEMINI RESPONSE:
    # Gemini receives the factual tool output and generates a comprehensive,
    # natural-language response for the user.
    try:
        final_response = client.models.generate_content(
            model=model_name,
            contents=contents,
        )
        final_text = (
            final_response.text.strip()
            if final_response and final_response.text
            else "No response generated."
        )
        print("\n==================================================")
        print("GEMINI FINAL RESPONSE")
        print("==================================================")
        print(f"\n{final_text}")
    except ClientError as e:
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            print(
                f"\nGEMINI API ERROR (429 Quota/Rate Limit Exhausted):\n"
                f"Free Tier limit reached. Unable to generate final response.\n"
                f"Details: {e}",
                file=sys.stderr,
            )
        else:
            print(f"\nGEMINI API ERROR ({e.code}):\n{e}", file=sys.stderr)
    except ServerError as e:
        if e.code == 503 or "UNAVAILABLE" in str(e):
            print(
                f"\nGEMINI API ERROR (503 Service Unavailable):\n"
                f"Gemini service is temporarily unavailable for final synthesis.\n"
                f"Details: {e}",
                file=sys.stderr,
            )
        else:
            print(f"\nGEMINI API SERVER ERROR ({e.code}):\n{e}", file=sys.stderr)
    except Exception as e:
        print(f"\nGEMINI API UNEXPECTED ERROR:\nFailed to generate final response: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Who created Java?"
    )
    asyncio.run(run_gemini_tool_agent(query))

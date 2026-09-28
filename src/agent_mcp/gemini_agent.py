"""src/agent_mcp/gemini_agent.py

STEP 5: Gemini Agent + Model Context Protocol (MCP) Integration

ARCHITECTURAL PRINCIPLE:
------------------------
Gemini does NOT directly call the MCP server.
The Python application remains the central orchestrator and bridge:

       USER REQUEST
            ↓
          GEMINI
            ↓
   (Tool decision: get_learning_topics?)
            ↓
       PYTHON AGENT  (Validates arguments & acts as bridge)
            ↓
        MCP CLIENT   (Stdio transport / JSON-RPC handshake)
            ↓
        MCP SERVER   (Registers tools & executes domain logic)
            ↓
   get_learning_topics()
            ↓
       TOOL RESULT
            ↓
       PYTHON AGENT  (Packages tool result into conversation history)
            ↓
          GEMINI     (Synthesizes final answer with domain context)
            ↓
      FINAL RESPONSE

Why this architecture is critical:
1. Security & Isolation: LLMs are non-deterministic and should never have direct,
   unmediated access to server processes, file systems, or network sockets.
2. Protocol Decoupling: Gemini reasons via JSON function definitions. The Python agent
   translates that intent into standardized MCP JSON-RPC protocol calls.
3. Observability & Control: Every tool call, argument, and result passes through the
   Python application where it can be logged, validated, and sandboxed.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any
from google import genai
from google.genai import types
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
# 1. API Key Retrieval (Safe: reads environment without exposing or writing keys)
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    """Read GEMINI_API_KEY from environment or Windows User registry fallback."""
    # Attempt to load from .env file if python-dotenv is present
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
# 2. Tool Definition for Gemini
# ---------------------------------------------------------------------------
# Expose get_learning_topics as a function declaration schema to Gemini.
# Note: Gemini receives only this JSON schema; it does NOT connect to the MCP server.
tool_declaration = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_learning_topics",
            description="Returns a structured list of important learning topics for a given technical subject.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "topic": types.Schema(
                        type="STRING",
                        description="The technical subject, framework, or language.",
                    ),
                },
                required=["topic"],
            ),
        )
    ]
)


# ---------------------------------------------------------------------------
# 3. MCP Client Bridge: Communicates with MCP Server over stdio
# ---------------------------------------------------------------------------
async def execute_mcp_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Connects to the MCP server via the MCP Client, calls the tool, and returns result.

    ARCHITECTURAL BRIDGE:
    The Python Agent invokes this function. The MCP client spawns the MCP server
    as a subprocess, manages JSON-RPC over stdio, and returns the result back to
    the Python agent. Gemini has zero knowledge of or access to this connection.
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
# 4. Gemini Agent Orchestration Workflow
# ---------------------------------------------------------------------------
async def run_gemini_agent(user_query: str) -> None:
    """Executes the Gemini decision-making and MCP tool calling lifecycle."""
    # Step A: Verify GEMINI_API_KEY exists before making any API call
    api_key = get_api_key()
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found.", file=sys.stderr)
        return

    print("GEMINI_API_KEY detected")

    print("==================================================")
    print("GEMINI AGENT + MCP")
    print("==================================================")
    print(f"\nUSER REQUEST:\n{user_query}")

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

    # Step B: Prepare conversation contents with user request
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )
    ]

    # Step C: Send request to Gemini with the MCP tool declaration
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[tool_declaration],
                temperature=0.0,
            ),
        )
    except Exception as e:
        print(f"\nGEMINI API ERROR:\n{e}", file=sys.stderr)
        return

    # Step D: Inspect Gemini's decision
    # Case 1: Gemini decides no tool is needed -> answers directly
    if not response.function_calls:
        print("\nGEMINI DECISION:\nNo tool required. Answering directly.")
        final_answer = response.text.strip() if response.text else "No response generated."
        print(f"\nFINAL RESPONSE:\n{final_answer}")
        return

    # Case 2: Gemini decides get_learning_topics is required
    call = response.function_calls[0]
    tool_name = call.name
    tool_args = call.args or {}

    print(f"\nGEMINI DECISION:\nTool call required: {tool_name}")
    print(f"\nSELECTED TOOL:\n{tool_name}")
    print(f"\nTOOL ARGUMENTS:\n{json.dumps(tool_args, indent=2)}")

    # Validate tool name
    if tool_name != "get_learning_topics":
        print(f"\nError: Unsupported tool '{tool_name}' requested by Gemini.", file=sys.stderr)
        return

    # Validate tool arguments safely
    if (
        not isinstance(tool_args, dict)
        or "topic" not in tool_args
        or not isinstance(tool_args["topic"], str)
        or not tool_args["topic"].strip()
    ):
        print(f"\nError: Malformed tool arguments received from Gemini: {tool_args}", file=sys.stderr)
        return

    # Step E: Python Agent bridges call to MCP Client -> MCP Server
    try:
        tool_result = await execute_mcp_tool(tool_name, tool_args)
    except Exception as e:
        print(f"\nMCP TOOL ERROR:\nFailed to execute tool on MCP server: {e}", file=sys.stderr)
        return

    print(f"\nMCP TOOL RESULT:\n{json.dumps(tool_result, indent=2)}")

    # Step F: Pass the MCP tool result back to Gemini to produce final natural language response
    if response.candidates and response.candidates[0].content:
        contents.append(response.candidates[0].content)

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
        print(f"\nFINAL RESPONSE:\n{final_text}")
    except Exception as e:
        print(f"\nGEMINI API ERROR:\nFailed to generate final response from tool result: {e}", file=sys.stderr)
        return


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Give me a Spring Boot learning roadmap"
    )
    asyncio.run(run_gemini_agent(query))

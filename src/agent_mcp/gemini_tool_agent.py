"""src/agent_mcp/gemini_tool_agent.py

STEP 7D: Gemini-Based MCP Tool Selection
----------------------------------------

ARCHITECTURAL PRINCIPLE:
------------------------
Gemini does NOT directly call the MCP server or external APIs.
The Python application serves as the bridge:

       USER REQUEST
            ↓
          GEMINI
            ↓
   (Gemini evaluates request & tool schemas:
    chooses 'get_learning_topics' OR 'search_wikipedia')
            ↓
       PYTHON AGENT  (Extracts & validates tool name & arguments)
            ↓
        MCP CLIENT   (Stdio transport / JSON-RPC handshake)
            ↓
        MCP SERVER   (Executes the selected tool)
            ↓
       TOOL RESULT
            ↓
       PYTHON AGENT  (Packages MCP result into conversation history)
            ↓
          GEMINI     (Synthesizes final answer with grounding data)
            ↓
      FINAL RESPONSE

EDUCATIONAL OVERVIEW:
--------------------
1. GEMINI TOOL DECLARATION:
   We declare tool schemas to Gemini using standard JSON schemas. Gemini learns
   what each tool does and what input parameters it accepts.
2. GEMINI TOOL SELECTION:
   When presented with the user request, Gemini's reasoning determines if a tool
   is needed, and if so, which tool and what arguments to supply.
3. MCP BRIDGE:
   The Python agent intercepts Gemini's function call, acts as the client, and
   forwards the call over stdio to the MCP Server.
4. MCP TOOL EXECUTION:
   The MCP Server executes the Python tool implementation and returns structured JSON.
5. SENDING THE TOOL RESULT BACK TO GEMINI:
   The tool result is sent back to Gemini in a function response turn so Gemini
   can incorporate real-world grounded facts into its final answer.
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
# 1. API Key Retrieval (Safe: reads environment without exposing or writing keys)
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
# 2. Tool Declarations for Gemini
# ---------------------------------------------------------------------------
# Define the schemas for the two available MCP tools.
# Gemini receives these definitions and decides which tool is appropriate.
tools_declaration = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_learning_topics",
            description="Returns a structured list of important learning topics/roadmap for a given technical subject.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "topic": types.Schema(
                        type="STRING",
                        description="The technical subject, framework, or language to generate learning topics for (e.g. 'Spring Boot', 'Python').",
                    ),
                },
                required=["topic"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_wikipedia",
            description="Search Wikipedia for an encyclopedic summary, definition, background, or history of a technology, concept, or person.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "topic": types.Schema(
                        type="STRING",
                        description="The technical topic, language, tool, or subject to search on Wikipedia (e.g. 'Java', 'Python').",
                    ),
                },
                required=["topic"],
            ),
        ),
    ]
)

SUPPORTED_TOOLS = ["get_learning_topics", "search_wikipedia"]


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
async def run_gemini_tool_agent(user_query: str) -> None:
    """Executes the Gemini decision-making and MCP tool calling lifecycle."""
    # Step A: Verify GEMINI_API_KEY exists before making any API call
    api_key = get_api_key()
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found.", file=sys.stderr)
        return

    print("==================================================")
    print("GEMINI + MCP TOOL SELECTION")
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

    # Step C: Send request to Gemini with both MCP tool declarations
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
            print(f"\nGEMINI API ERROR (429 Quota/Rate Limit):\nFree Tier limit reached. Please wait a moment before trying again.\nDetails: {e}", file=sys.stderr)
        elif e.code == 404:
            print(f"\nGEMINI API ERROR (404 Not Found):\nModel '{model_name}' was not found. Please verify GEMINI_MODEL setting.\nDetails: {e}", file=sys.stderr)
        else:
            print(f"\nGEMINI API ERROR ({e.code}):\n{e}", file=sys.stderr)
        return
    except ServerError as e:
        if e.code == 503 or "UNAVAILABLE" in str(e):
            print(f"\nGEMINI API ERROR (503 Service Unavailable):\nGemini API is temporarily overloaded or unavailable. Please try again shortly.\nDetails: {e}", file=sys.stderr)
        else:
            print(f"\nGEMINI API SERVER ERROR ({e.code}):\n{e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"\nGEMINI API UNEXPECTED ERROR:\n{e}", file=sys.stderr)
        return

    # Step D: Inspect Gemini's tool decision
    print("\n--------------------------------------------------")
    print("GEMINI TOOL DECISION")
    print("--------------------------------------------------")

    # Case 1: Gemini decides no tool is needed -> answers directly
    if not response.function_calls:
        print("\nGEMINI DID NOT REQUEST AN MCP TOOL")
        print("\n--------------------------------------------------")
        print("GEMINI FINAL RESPONSE")
        print("--------------------------------------------------")
        final_answer = response.text.strip() if response.text else "No response generated."
        print(f"\n{final_answer}")
        return

    # Case 2: Gemini selects one of the MCP tools
    call = response.function_calls[0]
    tool_name = call.name
    tool_args = call.args or {}

    print(f"\nSELECTED TOOL:\n{tool_name}")
    print(f"\nARGUMENTS:\n{json.dumps(tool_args, indent=2)}")

    # Validate tool name
    if tool_name not in SUPPORTED_TOOLS:
        print(f"\nError: Unsupported tool '{tool_name}' requested by Gemini.", file=sys.stderr)
        return

    # Validate arguments safely
    if (
        not isinstance(tool_args, dict)
        or "topic" not in tool_args
        or not isinstance(tool_args["topic"], str)
        or not tool_args["topic"].strip()
    ):
        print(f"\nError: Malformed tool arguments received from Gemini: {tool_args}", file=sys.stderr)
        return

    # Step E: Python Agent bridges call to MCP Client -> MCP Server
    print("\n--------------------------------------------------")
    print("MCP TOOL EXECUTION")
    print("--------------------------------------------------")
    print(f"\nTOOL:\n{tool_name}")

    try:
        tool_result = await execute_mcp_tool(tool_name, tool_args)
    except Exception as e:
        print(f"\nMCP TOOL ERROR:\nFailed to execute tool on MCP server: {e}", file=sys.stderr)
        return

    print(f"\nRESULT:\n{json.dumps(tool_result, indent=2)}")

    # Step F: Pass the MCP tool result back to Gemini to produce final response
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

    print("\n--------------------------------------------------")
    print("GEMINI FINAL RESPONSE")
    print("--------------------------------------------------")

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
        print(f"\n{final_text}")
    except ClientError as e:
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            print(f"\nGEMINI API ERROR (429 Quota/Rate Limit):\nFree Tier limit reached. Unable to generate final response.\nDetails: {e}", file=sys.stderr)
        else:
            print(f"\nGEMINI API ERROR ({e.code}):\n{e}", file=sys.stderr)
    except ServerError as e:
        if e.code == 503 or "UNAVAILABLE" in str(e):
            print(f"\nGEMINI API ERROR (503 Service Unavailable):\nGemini API is temporarily unavailable for final synthesis.\nDetails: {e}", file=sys.stderr)
        else:
            print(f"\nGEMINI API SERVER ERROR ({e.code}):\n{e}", file=sys.stderr)
    except Exception as e:
        print(f"\nGEMINI API UNEXPECTED ERROR:\nFailed to generate final response from tool result: {e}", file=sys.stderr)


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

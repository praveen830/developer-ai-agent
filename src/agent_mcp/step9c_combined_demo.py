"""src/agent_mcp/step9c_combined_demo.py

STEP 9C: Combined MCP Demonstration
-----------------------------------
Demonstrates the trifecta of Model Context Protocol (MCP) capabilities
orchestrated by a Python Agent and synthesized by Gemini:

1. MCP TOOL (Action):
   `search_wikipedia(topic="Spring Security")`
   Fetches dynamic, factual information from an external source.

2. MCP RESOURCE (Readable Data):
   `learning://spring-boot/roadmap`
   Supplies authoritative, read-only roadmap context.

3. MCP PROMPT (Instruction Template):
   `spring_boot_teacher(topic="Spring Boot Security")`
   Provides reusable pedagogical guidelines on how to explain concepts to beginners.

ARCHITECTURAL PRINCIPLE:
------------------------
Gemini NEVER directly connects to the MCP Server.
The Python Agent acts as the central bridge:

       User Request
            ↓
       Python Agent (Orchestrator)
            ↓
        MCP Client (Stdio transport / JSON-RPC protocol)
            ↓
        MCP Server
        ├── Resource: learning://spring-boot/roadmap
        ├── Prompt:   spring_boot_teacher("Spring Boot Security")
        └── Tool:     search_wikipedia("Spring Security")
            ↓
       Python Agent collects all 3 outputs
            ↓
          Gemini receives collected grounding context
            ↓
     === FINAL GEMINI ANSWER ===
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
# 2. Combined Workflow Orchestration
# ---------------------------------------------------------------------------
async def run_combined_demo(user_query: str) -> None:
    """Orchestrates MCP Resource reading, MCP Prompt retrieval, MCP Tool execution,
    and submits the gathered context to Gemini for final grounded synthesis.
    """
    print("==================================================")
    print("=== USER REQUEST ===")
    print("==================================================")
    print(f"\n{user_query}\n")
    sys.stdout.flush()

    # Step A: Connect to existing MCP Server over stdio
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["src/mcp_server/server.py"],
        env=dict(os.environ),
    )

    resource_text = ""
    prompt_text = ""
    tool_text = ""

    print("Connecting to MCP Server over stdio...")
    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # -----------------------------------------------------------
                # Capability 1: Read MCP Resource (learning://spring-boot/roadmap)
                # -----------------------------------------------------------
                try:
                    resource_uri = "learning://spring-boot/roadmap"
                    resource_response = await session.read_resource(resource_uri)
                    if resource_response.contents:
                        resource_text = resource_response.contents[0].text
                    else:
                        resource_text = "No content returned from resource."
                except Exception as e:
                    resource_text = f"Error reading MCP resource: {e}"

                # -----------------------------------------------------------
                # Capability 2: Retrieve MCP Prompt (spring_boot_teacher)
                # -----------------------------------------------------------
                try:
                    prompt_name = "spring_boot_teacher"
                    prompt_args = {"topic": "Spring Boot Security"}
                    prompt_response = await session.get_prompt(prompt_name, prompt_args)
                    if prompt_response.messages and hasattr(prompt_response.messages[0].content, "text"):
                        prompt_text = prompt_response.messages[0].content.text
                    else:
                        prompt_text = str(prompt_response.messages)
                except Exception as e:
                    prompt_text = f"Error retrieving MCP prompt: {e}"

                # -----------------------------------------------------------
                # Capability 3: Call MCP Tool (search_wikipedia)
                # -----------------------------------------------------------
                try:
                    tool_name = "search_wikipedia"
                    tool_args = {"topic": "Spring Security"}
                    tool_response = await session.call_tool(tool_name, tool_args)
                    tool_parts = []
                    for c in tool_response.content:
                        if c.type == "text":
                            try:
                                parsed = json.loads(c.text)
                                tool_parts.append(json.dumps(parsed, indent=2))
                            except json.JSONDecodeError:
                                tool_parts.append(c.text)
                        else:
                            tool_parts.append(str(c))
                    tool_text = "\n".join(tool_parts) if tool_parts else "{}"
                except Exception as e:
                    tool_text = f"Error executing MCP tool: {e}"

    except Exception as e:
        print(f"\nMCP SERVER CONNECTION ERROR:\nFailed to communicate with MCP server: {e}", file=sys.stderr)
        return

    # -----------------------------------------------------------------------
    # Step B: Print Separated Results
    # -----------------------------------------------------------------------
    print("==================================================")
    print("=== MCP RESOURCE ===")
    print("==================================================")
    print(f"\n{resource_text.strip()}\n")

    print("==================================================")
    print("=== MCP PROMPT ===")
    print("==================================================")
    print(f"\n{prompt_text.strip()}\n")

    print("==================================================")
    print("=== MCP TOOL ===")
    print("==================================================")
    print(f"\n{tool_text.strip()}\n")
    sys.stdout.flush()

    # -----------------------------------------------------------------------
    # Step C: Send Context to Gemini for Final Answer
    # -----------------------------------------------------------------------
    api_key = get_api_key()
    if not api_key:
        print(
            "CONFIGURATION NOTE: GEMINI_API_KEY environment variable is not set.\n"
            "MCP Resource, Prompt, and Tool were successfully retrieved, but Gemini "
            "cannot be called without GEMINI_API_KEY.",
            file=sys.stderr,
        )
        return

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

    # Combine the MCP Prompt (instructions), MCP Resource (roadmap), and MCP Tool result (facts)
    combined_instructions = (
        f"{prompt_text}\n\n"
        f"--- CONTEXT: AUTHORITATIVE ROADMAP (from MCP Resource 'learning://spring-boot/roadmap') ---\n"
        f"{resource_text}\n\n"
        f"--- CONTEXT: FACTUAL SUMMARY (from MCP Tool 'search_wikipedia') ---\n"
        f"{tool_text}\n\n"
        f"--- STUDENT REQUEST ---\n"
        f"{user_query}\n\n"
        f"Now produce the complete lesson adhering to all pedagogical guidelines in the prompt above, "
        f"grounding your explanation in the factual summary and showing clearly where Spring Security fits in the roadmap."
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=combined_instructions)],
        )
    ]

    print("==================================================")
    print("=== FINAL GEMINI ANSWER ===")
    print("==================================================")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
        )
        answer = response.text.strip() if response and response.text else "No response generated."
        print(f"\n{answer}\n")
    except ClientError as e:
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            print(
                f"\nGEMINI API NOTICE (429 Quota/Rate Limit Exhausted):\n"
                f"MCP Resource, Prompt, and Tool retrieval SUCCEEDED.\n"
                f"Only the final Gemini synthesis is unavailable due to Free Tier quota limits.\n"
                f"Details: {e}",
                file=sys.stderr,
            )
        else:
            print(f"\nGEMINI API ERROR ({e.code}): {e}", file=sys.stderr)
    except ServerError as e:
        if e.code == 503 or "UNAVAILABLE" in str(e):
            print(
                f"\nGEMINI API NOTICE (503 Service Unavailable):\n"
                f"MCP Resource, Prompt, and Tool retrieval SUCCEEDED.\n"
                f"Gemini API is temporarily experiencing high demand. Please try again shortly.\n"
                f"Details: {e}",
                file=sys.stderr,
            )
        else:
            print(f"\nGEMINI API SERVER ERROR ({e.code}): {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nGEMINI API UNEXPECTED ERROR: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    default_request = (
        "Teach me Spring Boot Security as a beginner, and explain where it fits in the Spring Boot learning roadmap."
    )
    query = sys.argv[1] if len(sys.argv) > 1 else default_request
    asyncio.run(run_combined_demo(query))

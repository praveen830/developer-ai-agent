"""src/mcp_client/client.py

STEP 7B: Python MCP Client - Multiple Tools
Connects to our local MCP server using standard I/O (stdio) transport,
discovers all registered tools, and executes both:
1. `get_learning_topics` with topic="Spring Boot"
2. `search_wikipedia` with topic="Java"
"""

import asyncio
import json
import os
import sys
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables (.env) if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure clean UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def format_mcp_result(call_result: Any) -> str:
    """Extracts and formats text or structured JSON from MCP call result content."""
    formatted_parts = []
    for content in call_result.content:
        if content.type == "text":
            try:
                parsed = json.loads(content.text)
                formatted_parts.append(json.dumps(parsed, indent=2))
            except json.JSONDecodeError:
                formatted_parts.append(content.text)
        else:
            formatted_parts.append(str(content))
    return "\n".join(formatted_parts) if formatted_parts else "{}"


async def run_mcp_client(profile_only: bool = False) -> None:
    """Connects to the MCP server, queries available tools, and invokes tools over stdio."""
    print("==================================================")
    print("MCP CLIENT - TOOL EXECUTION TEST")
    print("==================================================")

    # 1. StdioServerParameters defines how to spawn the MCP server subprocess
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["src/mcp_server/server.py"],
        env=dict(os.environ),
    )

    # 2. stdio_client launches the server process and creates JSON-RPC streams
    async with stdio_client(server_params) as (read_stream, write_stream):

        # 3. ClientSession manages protocol handshakes and tool requests
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("\nCONNECTED TO MCP SERVER")

            # 4. List all available tools exposed by the MCP server
            tools_response = await session.list_tools()
            tool_names = [tool.name for tool in tools_response.tools]

            print("\nAVAILABLE TOOLS:")
            for name in tool_names:
                print(f"- {name}")

            if not profile_only:
                # -------------------------------------------------------------------
                # Tool 1: get_learning_topics
                # -------------------------------------------------------------------
                tool1_name = "get_learning_topics"
                tool1_args = {"topic": "Spring Boot"}

                print(f"\n--------------------------------------------------")
                print(f"TOOL 1: {tool1_name}")
                print(f"--------------------------------------------------")
                print(f"\nARGUMENTS:\n{json.dumps(tool1_args, indent=2)}")

                if tool1_name in tool_names:
                    call_result1 = await session.call_tool(tool1_name, tool1_args)
                    print(f"\nRESULT:\n{format_mcp_result(call_result1)}")
                else:
                    print(f"\nError: Tool '{tool1_name}' not found on MCP server!", file=sys.stderr)

                # -------------------------------------------------------------------
                # Tool 2: search_wikipedia
                # -------------------------------------------------------------------
                tool2_name = "search_wikipedia"
                tool2_args = {"topic": "Java"}

                print(f"\n--------------------------------------------------")
                print(f"TOOL 2: {tool2_name}")
                print(f"--------------------------------------------------")
                print(f"\nARGUMENTS:\n{json.dumps(tool2_args, indent=2)}")

                if tool2_name in tool_names:
                    call_result2 = await session.call_tool(tool2_name, tool2_args)
                    print(f"\nRESULT:\n{format_mcp_result(call_result2)}")
                else:
                    print(f"\nError: Tool '{tool2_name}' not found on MCP server!", file=sys.stderr)

            # -------------------------------------------------------------------
            # Tool 3: get_student_profile (Spring Boot authenticated API call)
            # -------------------------------------------------------------------
            tool3_name = "get_student_profile"
            tool3_args: dict[str, Any] = {}

            print(f"\n--------------------------------------------------")
            print(f"TOOL 3: {tool3_name}")
            print(f"--------------------------------------------------")
            print(f"\nARGUMENTS:\n{json.dumps(tool3_args, indent=2)}")

            if tool3_name in tool_names:
                call_result3 = await session.call_tool(tool3_name, tool3_args)
                print(f"\nRESULT:\n{format_mcp_result(call_result3)}")
            else:
                print(f"\nError: Tool '{tool3_name}' not found on MCP server!", file=sys.stderr)

            print("\n==================================================")
            print("TEST COMPLETE")
            print("==================================================")


if __name__ == "__main__":
    profile_mode = "--profile-only" in sys.argv
    asyncio.run(run_mcp_client(profile_only=profile_mode))


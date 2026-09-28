"""src/agent_mcp/agent.py

STEP 7C: Agent Tool Selection with Multiple Model Context Protocol (MCP) Tools

This script demonstrates deterministic agent orchestration across multiple MCP tools:
USER -> AGENT -> UNDERSTAND -> SELECT TOOL -> MCP CLIENT -> MCP SERVER -> TOOL -> FINAL RESPONSE

ARCHITECTURAL PRINCIPLE:
------------------------
The Agent never directly executes external API requests (e.g. Wikipedia).
All capability is exposed as standardized MCP tools on the MCP server:
  1. `get_learning_topics`: Returns curated roadmap topics for a given technical subject.
  2. `search_wikipedia`: Returns introductory encyclopedic summaries from Wikipedia.

The Python application remains the orchestrator:
  Agent
    ↓
  MCP Client (Stdio transport / JSON-RPC protocol)
    ↓
  MCP Server (Hosts tools, validates parameters, and runs domain logic)
    ↓
  Selected MCP Tool (e.g., search_wikipedia or get_learning_topics)
    ↓
  Tool Result
    ↓
  Agent (Formats human-readable response)

NOTE ON DECISION-MAKING:
------------------------
This step uses deterministic keyword-based heuristics to demonstrate multi-tool selection
before integrating Gemini / LLM reasoning in the next step.
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


# ---------------------------------------------------------------------------
# 1. Understanding & Argument Extraction
# ---------------------------------------------------------------------------
def extract_topic(query: str) -> str:
    """Extracts the subject or technology name from the natural-language query."""
    q = query.strip()
    lower = q.lower()

    # Priority check for common multi-word or single-word technical subjects
    known_topics = [
        "spring boot",
        "artificial intelligence",
        "machine learning",
        "python",
        "java",
        "docker",
        "kubernetes",
        "javascript",
        "typescript",
        "react",
        "node.js",
        "golang",
        "rust",
    ]
    for topic in known_topics:
        if re.search(r"\b" + re.escape(topic) + r"\b", lower):
            return topic.title()

    # Fallback heuristic: strip common query phrases and punctuation
    cleaned = re.sub(
        r"(give\s+me|learning|roadmap|path|study\s+plan|curriculum|topics(\s+to\s+learn)?|for|about|who\s+(created|invented|made|founded)|what\s+(is|are|was)|history\s+of|explain|tell\s+me\s+about)",
        " ",
        q,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(a|an|the|of|in|to)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[?!.,;:]", " ", cleaned).strip()
    cleaned = " ".join(cleaned.split())
    return cleaned.title() if cleaned else "General"


def select_tool(user_query: str) -> tuple[str, str]:
    """Deterministically selects which MCP tool to invoke based on user intent keywords.

    Returns:
        (tool_name, intent_type)
    """
    query_lower = user_query.lower()

    # 1. Roadmap / curriculum triggers -> get_learning_topics
    roadmap_triggers = [
        "roadmap",
        "learning",
        "learn",
        "study",
        "topics",
        "curriculum",
        "what to learn",
        "how to learn",
    ]

    # 2. Encyclopedic / informational triggers -> search_wikipedia
    info_triggers = [
        "what is",
        "who created",
        "who invented",
        "history of",
        "explain",
        "tell me about",
        "summary",
        "overview",
        "definition",
    ]

    if any(trigger in query_lower for trigger in roadmap_triggers):
        return "get_learning_topics", "roadmap"
    elif any(trigger in query_lower for trigger in info_triggers):
        return "search_wikipedia", "info"
    else:
        # Default fallback to search_wikipedia for general informational queries
        return "search_wikipedia", "info"


# ---------------------------------------------------------------------------
# 2. MCP Client Bridge: Communicates with the MCP Server over stdio
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
# 3. Agent Orchestration Workflow
# ---------------------------------------------------------------------------
async def run_agent(user_query: str) -> None:
    """Runs the end-to-end agent workflow: understand, decide, execute, observe, respond."""
    print("==================================================")
    print("AGENT + MULTIPLE MCP TOOLS")
    print("==================================================")
    print(f"\nUSER REQUEST:\n{user_query}")

    # 1. UNDERSTAND REQUEST & EXTRACT TOPIC
    topic = extract_topic(user_query)
    tool_name, intent = select_tool(user_query)

    print("\n--------------------------------------------------")
    print("UNDERSTANDING")
    print("--------------------------------------------------")
    if intent == "roadmap":
        print(f"\nUser is asking for a learning roadmap for {topic}.")
    else:
        print(f"\nUser is asking for information about {topic}.")

    # 2. DETERMINISTIC AGENT DECISION
    tool_args = {"topic": topic}

    print("\n--------------------------------------------------")
    print("DETERMINISTIC AGENT DECISION")
    print("--------------------------------------------------")
    print(f"\nSELECTED TOOL:\n{tool_name}")
    print(f"\nTOOL ARGUMENTS:\n{json.dumps(tool_args, indent=2)}")

    # 3. CALL MCP TOOL THROUGH MCP CLIENT
    try:
        tool_result = await execute_mcp_tool(tool_name, tool_args)
    except Exception as e:
        tool_result = {"error": f"MCP execution failure: {str(e)}"}

    # 4. OBSERVE RESULT
    print("\n--------------------------------------------------")
    print("MCP TOOL RESULT")
    print("--------------------------------------------------")
    print(f"\n{json.dumps(tool_result, indent=2)}")

    # 5. FINAL RESPONSE FORMATTING
    print("\n--------------------------------------------------")
    print("FINAL RESPONSE")
    print("--------------------------------------------------\n")

    if tool_name == "get_learning_topics":
        topic_name = tool_result.get("topic", topic)
        topics_list = tool_result.get("topics", [])
        if topics_list:
            print(f"{topic_name} learning roadmap:\n")
            for i, item in enumerate(topics_list, start=1):
                print(f"{i}. {item}")
        else:
            print(f"No roadmap topics found for {topic_name}.")

    elif tool_name == "search_wikipedia":
        if tool_result.get("found") is False:
            print(tool_result.get("message") or tool_result.get("error") or "No article found.")
        else:
            title = tool_result.get("title", topic)
            summary = tool_result.get("summary", "")
            source = tool_result.get("source", "Wikipedia")
            print(f"Summary for {title} (Source: {source}):\n")
            print(summary)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
async def main() -> None:
    if len(sys.argv) > 1:
        # Run specific user-provided query
        await run_agent(sys.argv[1])
    else:
        # Run the 4 standard verification tests
        test_queries = [
            "Give me a Spring Boot learning roadmap",
            "Who created Java?",
            "What is Java?",
            "Give me a Python learning roadmap",
        ]
        for idx, query in enumerate(test_queries, start=1):
            await run_agent(query)
            if idx < len(test_queries):
                print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

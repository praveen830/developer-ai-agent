"""src/mcp_server/server.py

STEP 7A: Developer AI Agent - MCP Server with Multiple Tools
Exposes developer learning tools as standard Model Context Protocol (MCP) tools:
1. `get_learning_topics`: Returns curated roadmap topics for a given technical subject.
2. `search_wikipedia`: Queries Wikipedia MediaWiki API for encyclopedic summaries of technologies.

Built using the official MCP Python SDK.
"""

import asyncio
import json
import sys
from typing import Any
import requests
from mcp.server.mcpserver import MCPServer

# Initialize the MCP Server instance
mcp = MCPServer(name="developer-ai-agent")


# ---------------------------------------------------------------------------
# MCP Tool 1: get_learning_topics
# ---------------------------------------------------------------------------
@mcp.tool()
def get_learning_topics(topic: str) -> dict[str, Any]:
    """Retrieve structured learning topics/modules for a given technical topic.

    Args:
        topic: The technical subject, framework, or language (e.g., 'Spring Boot', 'Docker', 'Python').

    Returns:
        A dictionary containing the topic name and an ordered list of core learning topics.
    """
    knowledge_base = {
        "spring boot": [
            "Spring Core",
            "Dependency Injection",
            "REST APIs",
            "Spring Data JPA",
            "Spring Security",
        ],
        "docker": [
            "Container Fundamentals & Architecture",
            "Dockerfiles & Image Creation",
            "Docker Networking & Storage Volumes",
            "Docker Compose for Multi-Container Apps",
            "Container Security & Production Best Practices",
        ],
        "python": [
            "Python Syntax & Data Structures",
            "Object-Oriented Programming (OOP)",
            "Modules, Virtual Environments & Packaging",
            "File I/O & Working with APIs",
            "Testing & Asynchronous Programming (asyncio)",
        ],
    }

    key = topic.strip().lower()
    topics = knowledge_base.get(
        key,
        [
            f"{topic} Fundamentals & Core Concepts",
            f"{topic} Architecture & Workflow",
            f"Building Real-World Projects with {topic}",
            f"Testing & Debugging {topic} Applications",
            f"Advanced Patterns & Production Deployment",
        ],
    )

    return {
        "topic": topic,
        "topics": topics,
    }


# ---------------------------------------------------------------------------
# MCP Tool 2: search_wikipedia
# ---------------------------------------------------------------------------
@mcp.tool()
def search_wikipedia(topic: str) -> dict[str, Any]:
    """Search Wikipedia for a technical topic and retrieve a clean summary extract.

    WHAT THIS TOOL DOES:
    Accepts a subject query (e.g., 'Java', 'Docker', 'Spring Boot'), searches Wikipedia
    via the MediaWiki public API, and extracts the introductory summary in plain text.

    WHY THE WIKIPEDIA API IS CALLED:
    Provides grounding encyclopedic knowledge and definitions for concepts outside the
    agent's local knowledge base without needing web scrapers or full-page HTML parsing.

    HOW THE HTTP RESPONSE IS PROCESSED:
    1. Sends a search query (`list=search`) to find matching Wikipedia article titles.
    2. Selects the most relevant article (prioritizing programming/computing disambiguations).
    3. Fetches the article's intro extract (`prop=extracts&exintro=1&explaintext=1`).
    4. Formats and returns clean structured JSON to the MCP Client.

    SAFETY & ERROR HANDLING:
    - Handles empty / whitespace input safely.
    - Handles nonexistent topics with an informative message.
    - Catches network timeouts, HTTP errors, and JSON decode errors so the MCP server never crashes.
    """
    # 1. Handle empty or whitespace-only inputs gracefully
    clean_topic = topic.strip() if topic else ""
    if not clean_topic:
        return {
            "topic": topic,
            "found": False,
            "message": "Topic cannot be empty.",
            "source": "Wikipedia",
        }

    # 2. Wikipedia MediaWiki API configuration
    # A custom User-Agent is required per Wikipedia API etiquette to prevent 403 Forbidden errors.
    endpoint = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "developer-ai-agent/1.0 (learning MCP project; contact: developer@localhost)"
    }

    try:
        # Step A: Search for the most relevant article titles
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": clean_topic,
            "srlimit": 5,
            "format": "json",
        }
        search_response = requests.get(
            endpoint,
            params=search_params,
            headers=headers,
            timeout=10,
        )
        search_response.raise_for_status()
        search_data = search_response.json()

        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return {
                "topic": clean_topic,
                "found": False,
                "message": f"No Wikipedia article found for '{clean_topic}'.",
                "source": "Wikipedia",
            }

        titles = [item["title"] for item in search_results if "title" in item]
        topic_lower = clean_topic.lower()

        # Select the most appropriate title:
        # Prefer programming language / computing specific title if present in top hits (e.g. 'Java (programming language)')
        technical_matches = [
            t
            for t in titles
            if t.lower()
            in (
                f"{topic_lower} (programming language)",
                f"{topic_lower} (software)",
                f"{topic_lower} (framework)",
                f"{topic_lower} (computing)",
            )
        ]
        if technical_matches:
            chosen_title = technical_matches[0]
        else:
            exact_matches = [t for t in titles if t.lower() == topic_lower]
            if exact_matches:
                chosen_title = exact_matches[0]
            else:
                chosen_title = titles[0]

        # Step B: Fetch the introductory plain-text extract for the chosen title
        extract_params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "titles": chosen_title,
            "format": "json",
        }
        extract_response = requests.get(
            endpoint,
            params=extract_params,
            headers=headers,
            timeout=10,
        )
        extract_response.raise_for_status()
        extract_data = extract_response.json()

        pages = extract_data.get("query", {}).get("pages", {})
        if not pages:
            return {
                "topic": clean_topic,
                "found": False,
                "message": f"Could not retrieve extract for '{chosen_title}'.",
                "source": "Wikipedia",
            }

        page = next(iter(pages.values()))
        extract_text = page.get("extract", "").strip()

        if not extract_text:
            return {
                "topic": clean_topic,
                "title": chosen_title,
                "found": False,
                "message": f"No summary extract available for '{chosen_title}'.",
                "source": "Wikipedia",
            }

        # Step C: Return the structured result to the MCP Client
        return {
            "topic": clean_topic,
            "title": chosen_title,
            "summary": extract_text,
            "source": "Wikipedia",
        }

    except requests.exceptions.RequestException as e:
        # Gracefully handle network timeouts, DNS errors, or HTTP failures
        return {
            "topic": clean_topic,
            "found": False,
            "error": f"Wikipedia API request error: {str(e)}",
            "source": "Wikipedia",
        }
    except Exception as e:
        # Catch-all to ensure the MCP server never crashes
        return {
            "topic": clean_topic,
            "found": False,
            "error": f"Unexpected error while processing Wikipedia data: {str(e)}",
            "source": "Wikipedia",
        }


# ---------------------------------------------------------------------------
# Direct Test Helper (for running quick in-process verification without stdio)
# ---------------------------------------------------------------------------
async def _run_test() -> None:
    print("Testing MCP Server in-process...")
    tools = await mcp.list_tools()
    print(f"Discovered MCP Tools: {[tool.name for tool in tools]}")

    # Test 1: get_learning_topics
    test_topic = "Spring Boot"
    print(f"\n1. Invoking tool 'get_learning_topics' with topic='{test_topic}'...")
    result1 = await mcp.call_tool("get_learning_topics", {"topic": test_topic})
    print("MCP Tool 1 Returned Result:")
    for content in result1.content:
        if content.type == "text":
            print(content.text)

    # Test 2: search_wikipedia
    test_wiki = "Java"
    print(f"\n2. Invoking tool 'search_wikipedia' with topic='{test_wiki}'...")
    result2 = await mcp.call_tool("search_wikipedia", {"topic": test_wiki})
    print("MCP Tool 2 Returned Result:")
    for content in result2.content:
        if content.type == "text":
            print(content.text)


# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # If launched with --test argument, execute the local async verification
    if "--test" in sys.argv:
        asyncio.run(_run_test())
    else:
        # Standard MCP Server runner over stdio (for Inspector, Claude Desktop, Antigravity)
        mcp.run(transport="stdio")

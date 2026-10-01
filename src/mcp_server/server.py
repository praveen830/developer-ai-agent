"""src/mcp_server/server.py

STEP 7A: Developer AI Agent - MCP Server with Multiple Tools
Exposes developer learning tools as standard Model Context Protocol (MCP) tools:
1. `get_learning_topics`: Returns curated roadmap topics for a given technical subject.
2. `search_wikipedia`: Queries Wikipedia MediaWiki API for encyclopedic summaries of technologies.

Built using the official MCP Python SDK.
"""

import asyncio
import json
import os
import sys
from typing import Any
import requests
from mcp.server.mcpserver import MCPServer

# Load environment variables (.env) if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
# MCP Tool 3: get_student_profile
# ---------------------------------------------------------------------------
# ARCHITECTURE & SECURITY CONCEPTS:
#
# 1. MCP Tool:
#    In the Model Context Protocol (MCP), a "Tool" is an executable capability
#    that performs an action or dynamic data retrieval on behalf of the client/LLM.
#    Here, `get_student_profile` acts as a secure proxy function that bridges
#    our Python agent ecosystem with the backend Spring Boot microservice.
#
# 2. Spring Boot API Call:
#    The tool executes an HTTP GET request to `{CAREERAI_BASE_URL}/api/profile`.
#    The external Spring Boot server (`careerai-backend` running on port 8080)
#    serves this endpoint and returns the student's profile information as JSON.
#
# 3. Authorization Header:
#    The Spring Boot profile endpoint is secured with Spring Security JWT filter.
#    Clients must provide their credentials via the standard HTTP Authorization header:
#    `Authorization: Bearer <CAREERAI_AUTH_TOKEN>`.
#
# 4. Why the Token Must Remain Secret:
#    The JWT auth token is an active credential representing the authenticated user's
#    session and authority. If exposed:
#    - Unauthorized actors could impersonate the user and access or modify data.
#    - Leaking tokens into LLM prompts or chat logs could expose sensitive credentials.
#    Therefore:
#    - The token is read ONLY from environment variables (.env).
#    - The token is NEVER printed, logged, or displayed on the console.
#    - The token and Authorization header are NEVER returned in the tool output.
#    - The token is NEVER sent to Gemini or external LLM contexts.
# ---------------------------------------------------------------------------
@mcp.tool()
def get_student_profile() -> dict[str, Any]:
    """Retrieve the authenticated student's profile from the Spring Boot backend.

    Reads configuration from environment variables:
      - CAREERAI_BASE_URL: Base URL of the Spring Boot backend (e.g., http://localhost:8080)
      - CAREERAI_AUTH_TOKEN: Bearer JWT token issued by the backend

    Returns:
      A dictionary containing the parsed JSON profile data from the backend,
      or a structured error response if configuration or request fails.
    """
    base_url = os.environ.get("CAREERAI_BASE_URL", "").strip().rstrip("/")
    auth_token = os.environ.get("CAREERAI_AUTH_TOKEN", "").strip()

    # Step 1: Configuration Validation - fail clearly if either setting is missing
    if not base_url:
        return {
            "success": False,
            "error": (
                "Configuration error: CAREERAI_BASE_URL environment variable is missing or empty. "
                "Please configure CAREERAI_BASE_URL (e.g. http://localhost:8080) in your environment or .env file."
            ),
        }

    if not auth_token:
        return {
            "success": False,
            "error": (
                "Configuration error: CAREERAI_AUTH_TOKEN environment variable is missing or empty. "
                "Please configure CAREERAI_AUTH_TOKEN in your environment or .env file."
            ),
        }

    # Step 2: Build target endpoint and authentication headers
    target_url = f"{base_url}/api/profile"
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/json",
        "User-Agent": "developer-ai-agent/1.0",
    }

    # Step 3: Execute authenticated HTTP GET request with graceful error handling
    try:
        response = requests.get(target_url, headers=headers, timeout=10)

        # Parse JSON payload safely
        try:
            payload = response.json()
        except Exception:
            payload = response.text.strip() if response.text else None

        # Success case (HTTP 200-299)
        if 200 <= response.status_code < 300:
            if isinstance(payload, dict):
                # Preserve existing structure (e.g. ApiResponse wrapper or direct dict)
                return payload
            return {"success": True, "data": payload}

        # HTTP error status (e.g. 401 Unauthorized, 403 Forbidden, 404, 500)
        return {
            "success": False,
            "status_code": response.status_code,
            "error": f"Spring Boot API returned HTTP {response.status_code} ({response.reason}).",
            "response": payload,
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": (
                f"Connection error: Could not reach Spring Boot backend at {base_url}. "
                "Ensure the server is running on port 8080."
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Timeout error: Request to Spring Boot backend at {target_url} timed out.",
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"HTTP request failed: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error while retrieving student profile: {str(e)}",
        }


# ---------------------------------------------------------------------------
# MCP Tool 4: get_student_skills
# ---------------------------------------------------------------------------
@mcp.tool()
def get_student_skills() -> dict[str, Any]:
    """Retrieve the authenticated student's technical skill inventory from CareerAI.

    Reads configuration from environment variables:
      - CAREERAI_BASE_URL: Base URL of the Spring Boot backend
      - CAREERAI_AUTH_TOKEN: Bearer JWT token issued by the backend

    Returns:
      A dictionary containing the parsed JSON skill data from the backend,
      or a structured error response if configuration or request fails.
    """
    base_url = os.environ.get("CAREERAI_BASE_URL", "").strip().rstrip("/")
    auth_token = os.environ.get("CAREERAI_AUTH_TOKEN", "").strip()

    # Step 1: Configuration Validation
    if not base_url:
        return {
            "success": False,
            "error": (
                "Configuration error: CAREERAI_BASE_URL environment variable is missing or empty. "
                "Please configure CAREERAI_BASE_URL in your environment or .env file."
            ),
        }

    if not auth_token:
        return {
            "success": False,
            "error": (
                "Configuration error: CAREERAI_AUTH_TOKEN environment variable is missing or empty. "
                "Please configure CAREERAI_AUTH_TOKEN in your environment or .env file."
            ),
        }

    # Step 2: Build target endpoint and authentication headers
    target_url = f"{base_url}/api/skills"
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/json",
        "User-Agent": "developer-ai-agent/1.0",
    }

    # Step 3: Execute authenticated HTTP GET request
    try:
        response = requests.get(target_url, headers=headers, timeout=10)

        # Parse JSON payload safely
        try:
            payload = response.json()
        except Exception:
            payload = response.text.strip() if response.text else None

        # Success case (HTTP 200-299)
        if 200 <= response.status_code < 300:
            if isinstance(payload, dict):
                return payload
            return {"success": True, "data": payload}

        # HTTP error status
        return {
            "success": False,
            "status_code": response.status_code,
            "error": f"Spring Boot API returned HTTP {response.status_code} ({response.reason}).",
            "response": payload,
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": (
                f"Connection error: Could not reach Spring Boot backend at {base_url}."
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Timeout error: Request to Spring Boot backend at {target_url} timed out.",
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"HTTP request failed: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error while retrieving student skills: {str(e)}",
        }



# ---------------------------------------------------------------------------
# MCP Resource: learning://spring-boot/roadmap
# ---------------------------------------------------------------------------
# CONCEPT: Tool vs. Resource
# ---------------------------
# Tool: An active function that executes logic or performs actions based on arguments
#       (Tool = action, e.g., querying external APIs, searching data, dynamic calculation).
# Resource: Passive readable data exposed by a standardized URI
#       (Resource = readable data, e.g., documentation, roadmaps, static schema, reference texts).
# Clients read resources directly by URI without supplying execution arguments.
@mcp.resource("learning://spring-boot/roadmap")
def get_spring_boot_roadmap() -> str:
    """Provides a structured, beginner-friendly learning roadmap for Spring Boot."""
    return (
        "# Spring Boot Learning Roadmap\n\n"
        "A structured, beginner-friendly guide to mastering Spring Boot for enterprise backend development.\n\n"
        "1. Spring Core\n"
        "   - Inversion of Control (IoC) Container & ApplicationContext\n"
        "   - Bean Lifecycle, Scopes, and Configuration (@Configuration, @Bean)\n"
        "   - Component Scanning (@ComponentScan, @Component)\n\n"
        "2. Dependency Injection\n"
        "   - Constructor Injection (recommended) vs Setter and Field Injection\n"
        "   - Stereotype Annotations (@Component, @Service, @Repository, @Controller)\n"
        "   - Resolving Dependencies with @Autowired, @Primary, and @Qualifier\n\n"
        "3. REST APIs\n"
        "   - Building RESTful Web Services with Spring Web (@RestController, @RequestMapping)\n"
        "   - HTTP Request Mapping (@GetMapping, @PostMapping, @PutMapping, @DeleteMapping)\n"
        "   - Request & Response Handling (@PathVariable, @RequestParam, @RequestBody, ResponseEntity)\n"
        "   - Data Validation (@Valid, @NotNull) and Global Exception Handling (@ControllerAdvice)\n\n"
        "4. Spring Data JPA\n"
        "   - Object-Relational Mapping (ORM) with Hibernate entities (@Entity, @Table, @Id)\n"
        "   - Spring Data Repositories (CrudRepository, JpaRepository)\n"
        "   - Derived Query Methods and Custom JPQL Queries (@Query)\n"
        "   - Database Migrations with Flyway or Liquibase\n\n"
        "5. Spring Security\n"
        "   - SecurityFilterChain and Authentication / Authorization architecture\n"
        "   - Stateless Authentication with JSON Web Tokens (JWT)\n"
        "   - Role-Based Access Control (RBAC) and Method-Level Security (@PreAuthorize)\n"
        "   - Secure Password Hashing with BCryptPasswordEncoder\n"
    )


# ---------------------------------------------------------------------------
# MCP Prompt: spring_boot_teacher
# ---------------------------------------------------------------------------
# CONCEPT: Tool vs. Resource vs. Prompt
# -------------------------------------
# Tool: An executable function that performs actions or dynamic computation based on arguments (Action).
# Resource: Passive, readable data addressed by URI, like files or documentation (Readable Data).
# Prompt: A reusable, structured prompt template that guides an LLM on how to fulfill a specific task (Instruction Template).
@mcp.prompt(
    name="spring_boot_teacher",
    description="Provide reusable teaching instructions for explaining Spring Boot topics to a complete beginner.",
)
def spring_boot_teacher(topic: str = "Spring Boot") -> str:
    """Reusable prompt template instructing an LLM on how to teach a Spring Boot topic to a beginner."""
    return (
        f"You are an expert Spring Boot educator. Teach the topic '{topic}' to a complete beginner.\n\n"
        "Please follow these specific pedagogical guidelines:\n"
        f"1. Explain WHAT {topic} is in simple, plain language without jargon.\n"
        f"2. Explain WHY {topic} is used and the specific problems it solves in modern applications.\n"
        f"3. Explain HOW {topic} works step-by-step under the hood.\n"
        f"4. Use a relatable real-world analogy to make {topic} intuitive and memorable.\n"
        "5. Provide a clear, minimal code snippet or practical example illustrating the concept.\n"
        "6. Define any essential terminology or annotations associated with it.\n"
        "7. Avoid unnecessary advanced concepts, complex edge cases, or cognitive overload.\n"
        "8. Conclude with a concise 2-3 bullet point summary recap of key takeaways.\n"
    )


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

    # Test 3: MCP Resource
    print(f"\n3. Reading MCP Resource 'learning://spring-boot/roadmap'...")
    res_contents = await mcp.read_resource("learning://spring-boot/roadmap")
    print("MCP Resource Returned Content:")
    for res_part in res_contents:
        print(getattr(res_part, "content", res_part))

    # Test 4: MCP Prompt
    print(f"\n4. Retrieving MCP Prompt 'spring_boot_teacher' with topic='Dependency Injection'...")
    prompt_result = await mcp.get_prompt("spring_boot_teacher", {"topic": "Dependency Injection"})
    print("MCP Prompt Returned Result:")
    print(prompt_result)

    # Test 5: get_student_profile
    print("\n5. Invoking tool 'get_student_profile'...")
    result5 = await mcp.call_tool("get_student_profile", {})
    print("MCP Tool 3 (get_student_profile) Returned Result:")
    for content in result5.content:
        if content.type == "text":
            print(content.text)

    # Test 6: get_student_skills
    print("\n6. Invoking tool 'get_student_skills'...")
    result6 = await mcp.call_tool("get_student_skills", {})
    print("MCP Tool 4 (get_student_skills) Returned Result:")
    for content in result6.content:
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

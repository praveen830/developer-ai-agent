"""agent/agent.py

STEP 3: A single developer-learning AI Agent with one local tool.
Demonstrates the complete agentic tool-calling loop:
1. User prompt is sent to Gemini along with tool definitions.
2. Gemini reasons and decides whether to invoke the tool.
3. The local Python tool executes and returns structured data.
4. The tool result is returned to Gemini.
5. Gemini generates a structured, user-friendly final response.
"""

import json
import logging
import os
import sys
import time
from typing import Any
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

# Ensure UTF-8 output encoding on Windows consoles (prevents charmap UnicodeEncodeError)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress informational / AFC warning logs from the SDK to keep output clean
logging.getLogger("google_genai").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# 1. API Key Retrieval (Safe: reads environment without exposing or writing keys)
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    """Read GEMINI_API_KEY from environment or Windows User registry fallback."""
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
# 2. Local Tool Implementation
# ---------------------------------------------------------------------------
def get_learning_topics(topic: str) -> dict[str, Any]:
    """Retrieve structured learning topics/modules for a given technical topic."""
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


# Map tool name to the corresponding Python function
TOOL_REGISTRY = {
    "get_learning_topics": get_learning_topics,
}


# ---------------------------------------------------------------------------
# 3. Tool Declaration for Gemini
# ---------------------------------------------------------------------------
tool_declaration = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_learning_topics",
            description=(
                "Fetch the recommended ordered learning roadmap topics and core modules "
                "for a specified technical subject or framework."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "topic": types.Schema(
                        type="STRING",
                        description="The technical subject, framework, or language (e.g., 'Spring Boot', 'Docker', 'Python').",
                    ),
                },
                required=["topic"],
            ),
        )
    ]
)


# ---------------------------------------------------------------------------
# 4. Resilient Gemini API Caller (handles model quotas & transient retries)
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]


def call_gemini(
    client: genai.Client,
    contents: list[types.Content],
    tools: list[types.Tool] | None = None,
) -> Any:
    """Call Gemini with automatic fallback to another model if one hits rate/quota limits."""
    env_model = os.environ.get("GEMINI_MODEL")
    candidate_models = (
        [env_model] + [m for m in SUPPORTED_MODELS if m != env_model]
        if env_model
        else SUPPORTED_MODELS
    )

    last_error = None
    config = types.GenerateContentConfig(tools=tools) if tools else None

    for model in candidate_models:
        for attempt in range(2):
            try:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except ServerError:
                time.sleep(2 * (attempt + 1))
            except ClientError as e:
                # If quota exhausted (HTTP 429) for this model, try the next model
                if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
                    last_error = e
                    break
                raise

    if last_error:
        raise last_error
    return None


# ---------------------------------------------------------------------------
# 5. Agent Loop Implementation
# ---------------------------------------------------------------------------
def run_agent(user_query: str) -> None:
    """Executes the agent reasoning and tool-calling workflow."""
    api_key = get_api_key()
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("==================================================")
    print(f"USER REQUEST: {user_query}")
    print("==================================================")

    # Step 1 & 2: Send user request along with tool definitions
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )
    ]

    # Call Gemini with tool declaration
    response = call_gemini(client, contents, tools=[tool_declaration])
    if response is None:
        print("Failed to get response from Gemini API.", file=sys.stderr)
        return

    # Step 3: Check whether Gemini decided to call a tool or answer directly
    if response.function_calls:
        for call in response.function_calls:
            print("\nTOOL CALL")
            print(f"Tool Name : {call.name}")
            print(f"Arguments : {call.args}")

            tool_func = TOOL_REGISTRY.get(call.name)
            if not tool_func:
                print(f"Unknown tool '{call.name}' requested.", file=sys.stderr)
                return

            # Step 4: Execute the selected tool locally
            tool_output = tool_func(**call.args)

            print("\nTOOL RESULT")
            print(json.dumps(tool_output, indent=2))

            # Step 5 & 6: Send the tool result back to Gemini in conversation history
            contents.append(response.candidates[0].content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": tool_output},
                        )
                    ],
                )
            )

        # Step 7: Let Gemini generate the final answer using the tool result
        final_response = call_gemini(client, contents, tools=[tool_declaration])

        # Step 8: Print the final answer
        print("\nFINAL ANSWER")
        print("--------------------------------------------------")
        if final_response and final_response.text:
            print(final_response.text.strip())
        print("--------------------------------------------------")

    else:
        # Gemini decided no tool was needed; answered directly
        print("\n[DIRECT ANSWER - No tool call needed]")
        print("--------------------------------------------------")
        if response.text:
            print(response.text.strip())
        print("--------------------------------------------------")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_agent(sys.argv[1])
    else:
        print("\n>>> TEST 1: Request requiring tool")
        run_agent("Give me a Spring Boot learning roadmap.")

        print("\n" + "=" * 50 + "\n")

        print(">>> TEST 2: General question without roadmap request")
        run_agent("What is Java?")

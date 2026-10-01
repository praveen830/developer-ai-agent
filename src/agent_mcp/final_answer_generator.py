"""src/agent_mcp/final_answer_generator.py

STEP 10.3: Observation → Gemini → User-Friendly Final Answer
------------------------------------------------------------
Transforms raw MCP observations (or direct queries) into natural,
beginner-friendly, grounded final responses for the end user.

ARCHITECTURAL PRINCIPLE:
------------------------
Gemini plays two distinct roles in the agent lifecycle:
1. Gemini #1 (Step 10.1): Dynamic Capability Selection ("What capability should I use?")
2. Python Agent (Step 10.2): Validates decision & executes MCP capability over stdio.
3. Gemini #2 (Step 10.3): Synthesis ("I have the observation. How should I explain it to the user?")

       User Request
            ↓
       Gemini #1 (Step 10.1 Decision)
            ↓
       Python Agent Validation
            ↓
       MCP Client / Server (Step 10.2 Execution)
            ↓
       Observation / Result
            ↓
       Gemini #2 (Step 10.3 Synthesis)
            ↓
       User-Friendly Final Answer
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
# 2. Observation Formatting Helper
# ---------------------------------------------------------------------------
def format_observation(observation: Any) -> str:
    """Safely serializes various observation structures into clean text for prompt context."""
    if observation is None:
        return "None (Direct request without external observation)"

    if isinstance(observation, str):
        return observation.strip()

    if isinstance(observation, dict):
        # Pretty print structured JSON
        return json.dumps(observation, indent=2)

    return str(observation)


# ---------------------------------------------------------------------------
# 3. Final Answer Generator
# ---------------------------------------------------------------------------
def generate_final_answer(
    user_question: str,
    decision: dict[str, Any],
    observation: Any,
) -> str:
    """Invokes Gemini to synthesize the raw observation into a clear, grounded final answer.

    Args:
        user_question: The original question asked by the user.
        decision: The validated Step 10.1 decision dictionary.
        observation: The raw result produced by the Step 10.2 MCP executor.

    Returns:
        A user-friendly, natural-language explanation string.
    """
    api_key = get_api_key()
    if not api_key:
        return "CONFIGURATION ERROR: GEMINI_API_KEY environment variable is not set."

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    capability = decision.get("capability", "DIRECT") if isinstance(decision, dict) else "DIRECT"
    formatted_obs = format_observation(observation)

    # Tailored synthesis instructions based on capability type
    if capability == "DIRECT":
        guidance = (
            "This request does not require external MCP data. Answer the user's question "
            "directly, clearly, and concisely using your core programming knowledge."
        )
    elif capability == "RESOURCE":
        guidance = (
            "The observation contains authoritative reference data (e.g., a roadmap or curriculum). "
            "Present this information in an organized, beginner-friendly format for the user."
        )
    elif capability == "PROMPT":
        guidance = (
            "The observation contains pedagogical teaching instructions. Follow those instructions "
            "strictly to teach the concept to the user in a clear, beginner-friendly manner."
        )
    elif capability == "MULTI":
        guidance = (
            "The observation contains multiple context sources (e.g., teaching instructions and roadmap). "
            "Synthesize them together into a comprehensive, cohesive lesson."
        )
    else:  # TOOL
        guidance = (
            "The observation contains factual/encyclopedic lookup data. Ground your answer in "
            "this factual data. Do not hallucinate or invent unsupported details."
        )

    prompt = f"""You are a helpful, beginner-friendly AI assistant.

ORIGINAL USER QUESTION:
{user_question}

CAPABILITY DECISION:
{json.dumps(decision, indent=2) if isinstance(decision, dict) else str(decision)}

OBSERVATION:
{formatted_obs}

SYNTHESIS GUIDELINES:
1. Answer the original user question directly and thoroughly.
2. {guidance}
3. Do not expose internal architectural details (e.g., do NOT mention 'Step 10.1', 'Step 10.2', 'Step 10.3', 'MCP Client', 'MCP Server', 'JSON schema', or 'Tool call' in your response).
4. Keep the explanation natural, accessible, and well-structured using clean markdown.
5. If the observation indicates an error or is incomplete, explain that honestly rather than making up information.
"""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,  # Grounded yet natural synthesis
            ),
        )
        if response and response.text:
            return response.text.strip()
        return "No response generated by model."

    except ClientError as e:
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            return f"Status: Gemini temporarily unavailable / quota limited (429 RESOURCE_EXHAUSTED: {e})"
        return f"Status: Gemini client error ({e.code}): {e}"

    except ServerError as e:
        if e.code == 503 or "UNAVAILABLE" in str(e):
            return f"Status: Gemini temporarily unavailable / quota limited (503 UNAVAILABLE: {e})"
        return f"Status: Gemini server error ({e.code}): {e}"

    except Exception as e:
        return f"Status: Gemini unexpected error: {e}"


# ---------------------------------------------------------------------------
# 4. Controlled Test Runner
# ---------------------------------------------------------------------------
async def run_controlled_tests() -> None:
    """Executes the 3 required conceptual tests without unnecessary loops."""
    test_cases = [
        # Example 1: TOOL (Factual lookup)
        {
            "id": "Example 1 (TOOL)",
            "question": "Who created Java?",
            "decision": {
                "capability": "TOOL",
                "reason": "Requires factual and historical information about Java from Wikipedia.",
                "recommended_capabilities": ["search_wikipedia"],
            },
            "observation": {
                "topic": "Java",
                "title": "Java (programming language)",
                "summary": "Java was designed by James Gosling at Sun Microsystems. It was released in May 1995 as a core component of Sun's Java platform.",
                "source": "Wikipedia",
            },
        },
        # Example 2: RESOURCE (Roadmap)
        {
            "id": "Example 2 (RESOURCE)",
            "question": "Show me the Spring Boot learning roadmap.",
            "decision": {
                "capability": "RESOURCE",
                "reason": "Requires existing authoritative Spring Boot roadmap document.",
                "recommended_capabilities": ["learning://spring-boot/roadmap"],
            },
            "observation": (
                "# Spring Boot Learning Roadmap\n\n"
                "1. Spring Core (IoC, ApplicationContext, Beans)\n"
                "2. Dependency Injection (Constructor Injection, Stereotypes)\n"
                "3. REST APIs (Spring Web, Controllers, DTOs)\n"
                "4. Spring Data JPA (Hibernate, Repositories)\n"
                "5. Spring Security (Authentication, JWT, RBAC)"
            ),
        },
        # Example 3: DIRECT (General conceptual question)
        {
            "id": "Example 3 (DIRECT)",
            "question": "What is a variable?",
            "decision": {
                "capability": "DIRECT",
                "reason": "Basic conceptual question that does not require external tools or data.",
                "recommended_capabilities": [],
            },
            "observation": None,
        },
    ]

    for idx, tc in enumerate(test_cases, start=1):
        print("========================================")
        print("STEP 10.3 - FINAL ANSWER GENERATOR")
        print("========================================")
        print(f"\nUser Question:\n{tc['question']}")
        print(f"\nDecision:\n{json.dumps(tc['decision'], indent=2)}")
        print(f"\nObservation:\n{format_observation(tc['observation'])}")

        # Polite 2-second pause between requests to respect Free Tier limits
        if idx > 1:
            await asyncio.sleep(2)

        answer = generate_final_answer(
            user_question=tc["question"],
            decision=tc["decision"],
            observation=tc["observation"],
        )

        print(f"\nGemini Final Answer:\n{answer}\n")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run_controlled_tests())

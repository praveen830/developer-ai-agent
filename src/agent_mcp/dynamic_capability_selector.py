"""src/agent_mcp/dynamic_capability_selector.py

STEP 10.1: Dynamic MCP Capability Selection
--------------------------------------------

ARCHITECTURAL PRINCIPLE:
------------------------
This module is strictly focused on CLASSIFICATION & DECISION-MAKING:
Determining which MCP capability (Tool, Resource, Prompt, Multi, or Direct)
is appropriate for a given user query without executing the capability.

       User Request
            ↓
          Gemini (Evaluates intent against MCP Capability Schemas)
            ↓
     Classification JSON
            ↓
       Python Agent (Validates schema, categories, & capability identifiers)
            ↓
     Validated Decision

CAPABILITY CATEGORIES:
----------------------
1. TOOL:
   The request requires an executable action or external/factual lookup.
   Available: `get_learning_topics`, `search_wikipedia`.

2. RESOURCE:
   The request asks for existing structured reference/roadmap data.
   Available: `learning://spring-boot/roadmap`.

3. PROMPT:
   The request requires beginner-friendly teaching/explanation instructions.
   Available: `spring_boot_teacher`.

4. MULTI:
   The request requires multiple MCP capabilities combined.

5. DIRECT:
   The request can be answered directly by the LLM without using MCP.
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
# Whitelist Constants for Python Validation
# ---------------------------------------------------------------------------
VALID_CATEGORIES = {"TOOL", "RESOURCE", "PROMPT", "MULTI", "DIRECT"}

ALLOWED_CAPABILITIES = {
    "get_learning_topics",
    "search_wikipedia",
    "get_student_profile",
    "get_student_skills",
    "learning://spring-boot/roadmap",
    "spring_boot_teacher",
}

# System prompt defining capabilities and strict JSON structure for Gemini
SYSTEM_PROMPT = """You are an expert MCP (Model Context Protocol) Capability Classifier.
Analyze the user request and determine which MCP capability or capabilities are required.

AVAILABLE MCP CAPABILITIES:
1. TOOLS (Executable Actions / Factual Lookups):
   - get_learning_topics: Retrieve structured Spring Boot learning topics.
   - search_wikipedia: Retrieve factual/encyclopedic information from Wikipedia.
   - get_student_profile: Retrieve the authenticated student's profile from CareerAI (academic information, degree, branch, college, current year, CGPA, graduation year, profile completion). Requires NO arguments.
   - get_student_skills: Retrieve the authenticated student's technical skill inventory from CareerAI (skill name, proficiency, experience, last used). Requires NO arguments.

2. RESOURCE (Authoritative Readable Data):
   - learning://spring-boot/roadmap: Read the existing authoritative Spring Boot learning roadmap.

3. PROMPT (Pedagogical Teaching Instructions):
   - spring_boot_teacher: Provide reusable teaching guidelines for explaining technical topics to beginners.

CLASSIFICATION RULES:
- "TOOL": The request asks for factual/historical lookup or information about technologies, creators, or topics (e.g. "Who created Java?", "What is Docker?") which should be verified via "search_wikipedia", dynamic topics via "get_learning_topics", asks about the authenticated user's student profile/academic information (use "get_student_profile"), OR asks about the student's current skills, technologies the student knows, skill proficiency, skill experience, or what technologies are already known (use "get_student_skills").
- "RESOURCE": The request specifically asks for the existing Spring Boot learning roadmap document (use "learning://spring-boot/roadmap").
- "PROMPT": The request asks to be taught or explained a topic in a beginner-friendly way (use "spring_boot_teacher").
- "MULTI": The request requires multiple MCP capabilities (e.g., teaching instructions + roadmap context or factual lookup).
- "DIRECT": Basic conceptual questions that do not require external verification or specialized tools (e.g. "What is a variable?").

You MUST respond ONLY with a strict JSON object following this exact schema:
{
  "capability": "TOOL" | "RESOURCE" | "PROMPT" | "MULTI" | "DIRECT",
  "reason": "concise explanation of why this capability was chosen",
  "recommended_capabilities": ["capability_identifier", ...]
}

For DIRECT: "recommended_capabilities" must be an empty list [].
"""


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
# 2. Python Validation Layer
# ---------------------------------------------------------------------------
def validate_decision(decision: dict[str, Any]) -> tuple[bool, str]:
    """Strictly validates the JSON decision structure returned by Gemini."""
    if not isinstance(decision, dict):
        return False, "Output is not a valid dictionary."

    capability = decision.get("capability")
    if capability not in VALID_CATEGORIES:
        return False, f"Invalid category '{capability}'. Must be one of {sorted(VALID_CATEGORIES)}."

    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False, "Field 'reason' must be a non-empty string."

    recommended = decision.get("recommended_capabilities")
    if not isinstance(recommended, list):
        return False, "Field 'recommended_capabilities' must be a list."

    for item in recommended:
        if item not in ALLOWED_CAPABILITIES:
            return False, f"Unknown capability '{item}'. Must be from {sorted(ALLOWED_CAPABILITIES)}."

    if capability == "DIRECT" and len(recommended) != 0:
        return False, f"DIRECT category requires empty recommended_capabilities, found: {recommended}."

    if capability in {"TOOL", "RESOURCE", "PROMPT"} and len(recommended) != 1:
        return False, f"Category '{capability}' expects exactly 1 recommended capability, found: {recommended}."

    if capability == "MULTI" and len(recommended) < 2:
        return False, f"MULTI category requires 2 or more recommended capabilities, found: {recommended}."

    return True, "Decision conforms to all validation rules."


# ---------------------------------------------------------------------------
# 3. Dynamic Capability Classification Function
# ---------------------------------------------------------------------------
def classify_request(client: genai.Client, model_name: str, user_query: str) -> dict[str, Any] | None:
    """Invokes Gemini with structured JSON output to classify the user request."""
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"{SYSTEM_PROMPT}\n\nUSER REQUEST:\n\"{user_query}\"")
            ],
        )
    ]

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,  # Deterministic categorization
            ),
        )
        if response and response.text:
            return json.loads(response.text)
        return None
    except ClientError as e:
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            print(f"GEMINI NOTICE (429 Quota Exceeded): {e}", file=sys.stderr)
        else:
            print(f"GEMINI CLIENT ERROR ({e.code}): {e}", file=sys.stderr)
        return None
    except ServerError as e:
        if e.code == 503 or "UNAVAILABLE" in str(e):
            print(f"GEMINI NOTICE (503 Service Unavailable): {e}", file=sys.stderr)
        else:
            print(f"GEMINI SERVER ERROR ({e.code}): {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"GEMINI UNEXPECTED ERROR: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 4. Controlled Test Runner
# ---------------------------------------------------------------------------
async def run_tests() -> None:
    """Runs the 5 required test cases in sequence."""
    api_key = get_api_key()
    if not api_key:
        print("CONFIGURATION ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        return

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    test_queries = [
        "Who created Java?",
        "Show me the Spring Boot learning roadmap.",
        "Teach me Dependency Injection like a beginner.",
        "Teach me Spring Security and explain where it fits in the Spring Boot roadmap.",
        "What is a variable?",
    ]

    for idx, query in enumerate(test_queries, start=1):
        print("==================================================")
        print(f"=== TEST CASE {idx} ===")
        print("==================================================")
        print("=== USER REQUEST ===")
        print(f"{query}\n")

        # Brief polite pause between requests to prevent Free Tier burst throttling
        if idx > 1:
            await asyncio.sleep(2)

        decision = classify_request(client, model_name, query)

        print("=== GEMINI DECISION ===")
        if decision:
            print(json.dumps(decision, indent=2))
        else:
            print("Gemini decision could not be retrieved due to API quota/availability.")

        print("\n=== VALIDATION ===")
        if decision:
            is_valid, validation_msg = validate_decision(decision)
            if is_valid:
                print(f"PASS ({validation_msg})")
            else:
                print(f"FAIL ({validation_msg})")
        else:
            print("FAIL (No decision returned to validate)")

        print()


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run single custom query
        single_query = sys.argv[1]
        api_key = get_api_key()
        if not api_key:
            print("CONFIGURATION ERROR: GEMINI_API_KEY not found.", file=sys.stderr)
            sys.exit(1)
        cli_client = genai.Client(api_key=api_key)
        cli_model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

        print("=== USER REQUEST ===")
        print(single_query)
        res = classify_request(cli_client, cli_model, single_query)
        print("\n=== GEMINI DECISION ===")
        print(json.dumps(res, indent=2) if res else "None")
        print("\n=== VALIDATION ===")
        if res:
            valid, msg = validate_decision(res)
            print("PASS" if valid else f"FAIL ({msg})")
        else:
            print("FAIL (No response)")
    else:
        # Run the 5 standard verification tests
        asyncio.run(run_tests())

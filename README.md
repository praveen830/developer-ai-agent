# developer-ai-agent

A single-agent AI system built with **Python**, **Gemini**, and **MCP (Model Context Protocol)** to help developers create structured learning roadmaps for technical topics.

## Project Structure

```text
developer-ai-agent/
├── .env.example        # Environment variable template (Gemini API key)
├── .gitignore          # Standard Python & environment ignore rules
├── requirements.txt    # Minimal dependencies (google-genai, mcp, python-dotenv)
├── README.md           # Project documentation
├── roadmaps/           # Destination folder for generated roadmaps
└── src/
    ├── __init__.py
    ├── agent/          # Single AI Agent logic (prompt design, reasoning, tool execution)
    ├── mcp_server/     # MCP Server exposing roadmap tools (schema, handlers, resources)
    └── utils/          # Config loading and formatting helpers
```

## Goals

1. **Step-by-step learning**: Understand every component (Prompting, Agent Loop, MCP Tools, and Integration).
2. **Minimal & Focused**: No vector DBs, no multi-agent complexity, and no heavy frameworks.
3. **MCP Standards**: Expose roadmap capabilities via Model Context Protocol tools.

# AgenticAI

A hands-on project to build your first AI agent using **LangChain**, **Claude Sonnet 4**, and **Open-Meteo** — with every decision explained step by step.

## What This Agent Does

- Answers weather questions in natural language ("What's the weather in Chennai?")
- Fetches **live weather data** from Open-Meteo API (free, no API key needed)
- **Remembers conversation context** ("Which city was hotter?" works across turns)
- **Validates input** with Pydantic (format) + LLM (is it a real city?)
- **Personalizes responses** per user (Celsius vs. Fahrenheit) via ToolRuntime

## Quick Start

```bash
# Clone
git clone https://github.com/shaikmaricar/AgenticAI.git
cd AgenticAI

# Install dependencies
uv sync

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"          # Linux/Mac
$env:ANTHROPIC_API_KEY = "your-key-here"           # Windows PowerShell

# Run
uv run python main.py
```

## Sample Session

```
Weather Agent (type 'quit' to exit)
==================================================

You: What's the weather in Chennai?
Agent: The current weather in Chennai is 27.3°C, Partly cloudy,
       Humidity: 90%, Wind: 4.2 km/h

You: How about Tokyo?
Agent: The current weather in Tokyo is 13.3°C, Mainly clear,
       Humidity: 94%, Wind: 2.3 km/h

You: Which city is hotter?
Agent: Chennai is hotter at 27.3°C compared to Tokyo's 13.3°C —
       about 14°C warmer!
```

## Tech Stack

| Library | Purpose |
|---|---|
| [LangChain](https://python.langchain.com/) | Agent framework |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Agent runtime (ReAct loop, state, checkpointing) |
| [Claude Sonnet 4](https://docs.anthropic.com/) | LLM brain |
| [Open-Meteo](https://open-meteo.com/) | Free weather API (no key required) |
| [Pydantic](https://docs.pydantic.dev/) | Input validation |
| [httpx](https://www.python-httpx.org/) | HTTP client |

## Key Concepts Covered

```
User Input
  │
  ├─ Pydantic Validation (format: length, type)
  │
  ├─ LLM Validation (semantic: is it a real city?)
  │
  ├─ ReAct Loop (reason → act → observe → respond)
  │     │
  │     ├─ Tools with ToolRuntime (per-user context)
  │     ├─ recursion_limit (infinite loop safety)
  │     └─ InMemorySaver (conversation memory)
  │
  └─ Response to User
```

## Detailed Article

For a complete walkthrough explaining **why** each piece exists, every gotcha we hit, and when to use one approach over another:

[How to Create Your First AI Agent? An Easy to Understand Detailed Article](How-to-create-your-first-AI-agent.md)

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Anthropic API key](https://console.anthropic.com/)

## License

MIT

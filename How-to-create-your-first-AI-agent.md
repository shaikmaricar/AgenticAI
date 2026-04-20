# How to Create Your First AI Agent? An Easy to Understand Detailed Article

*A hands-on guide for beginners to build a weather agent with LangChain, Claude, and real-time APIs — with every decision explained.*

---

## What You'll Build

By the end of this article, you'll have a fully functional AI weather agent that:

- Accepts natural language questions like "What's the weather in Chennai?"
- Calls a real weather API to get live data
- Remembers your conversation ("Which city was hotter?")
- Validates user input before processing
- Personalizes responses per user (Celsius vs. Fahrenheit)

More importantly, you'll understand **why** each piece exists and **when** to use one approach over another.

---

## Prerequisites

- Python 3.12+
- An Anthropic API key (get one at [console.anthropic.com](https://console.anthropic.com))
- Basic Python knowledge

---

## Table of Contents

1. [What Is Agentic AI?](#1-what-is-agentic-ai)
2. [Agentic AI vs. Service API](#2-agentic-ai-vs-service-api)
3. [Project Setup](#3-project-setup)
4. [Step 1: The Simplest Agent](#4-step-1-the-simplest-agent)
5. [Step 2: Choosing the Right Import](#5-step-2-choosing-the-right-import)
6. [Step 3: String Model vs. ChatAnthropic Object](#6-step-3-string-model-vs-chatanthropic-object)
7. [Step 4: Preventing Infinite Loops](#7-step-4-preventing-infinite-loops)
8. [Step 5: Input Validation with Pydantic](#8-step-5-input-validation-with-pydantic)
9. [Step 6: LLM-Powered City Validation](#9-step-6-llm-powered-city-validation)
10. [Step 7: ToolRuntime — Giving Tools Context](#10-step-7-toolruntime--giving-tools-context)
11. [Step 8: Configuring the LLM](#11-step-8-configuring-the-llm)
12. [Step 9: Multi-Turn Conversation with InMemorySaver](#12-step-9-multi-turn-conversation-with-inmemorysaver)
13. [Step 10: Real Weather Data with Open-Meteo](#13-step-10-real-weather-data-with-open-meteo)
14. [The Final Code](#14-the-final-code)
15. [Key Concepts Recap](#15-key-concepts-recap)
16. [Where Agentic AI Is Used Today](#16-where-agentic-ai-is-used-today)
17. [Common Gotchas](#17-common-gotchas)

---

## 1. What Is Agentic AI?

Traditional AI takes input and produces output. **Agentic AI** can *reason*, *decide*, and *act*.

Think of it as the difference between a calculator and a human assistant:

```
Calculator (Traditional AI):
  Input:  2 + 2
  Output: 4

Human Assistant (Agentic AI):
  Input:  "Is it going to rain in Chennai? Should I carry an umbrella?"
  Think:  "I need to check the weather for Chennai first..."
  Act:    [Calls weather API]
  Observe: "27°C, Partly Cloudy, 10% rain chance"
  Think:  "Low rain chance, no umbrella needed"
  Output: "It's 27°C and partly cloudy in Chennai.
           Only 10% chance of rain — no umbrella needed!"
```

This think-act-observe loop is called **ReAct** (Reasoning + Acting), and it's the core pattern behind most AI agents today.

```
         ┌──────────┐
         │  USER    │
         │  INPUT   │
         └────┬─────┘
              │
              ▼
    ┌─────────────────┐
    │    REASON        │◄──────────────┐
    │ "What should I   │               │
    │  do with this?"  │               │
    └────────┬─────────┘               │
             │                         │
             ▼                         │
    ┌─────────────────┐               │
    │     ACT          │               │
    │ Call a tool,     │               │
    │ search, compute  │               │
    └────────┬─────────┘               │
             │                         │
             ▼                         │
    ┌─────────────────┐               │
    │    OBSERVE       │───────────────┘
    │ Did I get what   │   (loop back if
    │ I needed?        │    more work needed)
    └────────┬─────────┘
             │ (done)
             ▼
    ┌─────────────────┐
    │    RESPOND       │
    │ Final answer     │
    │ to user          │
    └─────────────────┘
```

---

## 2. Agentic AI vs. Service API

Before building, it's worth understanding *when* you'd use an agent vs. a simple API call.

**Service API approach** — you write all the routing logic:

```python
# You must handle every case manually
def handle_request(user_input):
    if "weather" in user_input:
        city = extract_city(user_input)        # you write this parser
        if is_valid_city(city):                # you maintain a city list
            data = call_weather_api(city)       # you handle API errors
            return format_response(data)        # you format the output
        else:
            return "Invalid city"
    elif "compare" in user_input:
        cities = extract_multiple_cities(user_input)  # more parsing
        results = [call_weather_api(c) for c in cities]
        return compare_results(results)               # more formatting
    else:
        return "I don't understand"
```

**Agentic AI approach** — you define tools, the agent handles everything:

```python
# You just define what tools exist
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return call_open_meteo(city)

# The agent handles understanding, routing, calling, formatting
agent = create_agent(model=llm, tools=[get_weather])
```

Here's when to use which:

| | Service API | Agentic AI |
|---|---|---|
| **Decision making** | You write all the if/else logic | Agent decides what to do |
| **Tool selection** | You hardcode which API to call | Agent picks the right tool |
| **Multi-step tasks** | You chain API calls manually | Agent chains automatically |
| **Input flexibility** | Strict format (`city=Chennai`) | Natural language ("How's Chennai?") |
| **Speed** | Milliseconds | Seconds (LLM inference time) |
| **Cost** | Cheap (API calls only) | Expensive (LLM tokens per request) |
| **Determinism** | Same input = same output | Can vary between runs |

**Rule of thumb:** Use agents when the input is unpredictable (natural language) and the workflow requires reasoning. Use APIs when speed, cost, and determinism matter.

---

## 3. Project Setup

We'll use [uv](https://docs.astral.sh/uv/) for dependency management — it's fast and handles Python versions automatically.

```bash
# Create project
mkdir langchain-practice && cd langchain-practice
uv init

# Add dependencies
uv add langchain langchain-anthropic langgraph httpx
```

This generates a `pyproject.toml`:

```toml
[project]
name = "langchain-practice"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langchain>=1.2.15",
    "langchain-anthropic>=1.4.1",
    "langgraph>=0.4.1",
    "httpx>=0.28.0",
]
```

**Why these libraries?**

| Library | Purpose |
|---|---|
| `langchain` | Core framework — agents, tools, prompts |
| `langchain-anthropic` | Claude model integration |
| `langgraph` | Agent runtime (ReAct loop, state management) |
| `httpx` | HTTP client for calling weather APIs |

Set your API key:

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="your-key-here"

# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your-key-here"

# Windows — persist across sessions
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "your-key-here", "User")
```

---

## 4. Step 1: The Simplest Agent

Let's start with the absolute minimum — matching the official LangChain docs example:

```python
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in sf?"}]}
)
```

That's it. Three components:

1. **Model** — the LLM brain (`"claude-sonnet-4-6"`)
2. **Tool** — a function the agent can call (`get_weather`)
3. **Prompt** — instructions for the agent's behavior

> **Note:** The tool returns fake data — `"It's always sunny in {city}!"` for every city. This is intentional. The docs example demonstrates the *pattern*, not real weather data. We'll add a real API later.

### How the agent processes this:

```
User: "What is the weather in sf?"
  │
  ▼
Agent REASONS: "User wants weather for SF. I have a get_weather tool."
  │
  ▼
Agent ACTS: get_weather("sf")  →  "It's always sunny in sf!"
  │
  ▼
Agent OBSERVES: Got the result.
  │
  ▼
Agent RESPONDS: "The weather in San Francisco is sunny!"
```

---

## 5. Step 2: Choosing the Right Import

### The Deprecation Story

When we first wrote the agent, we imported from LangGraph:

```python
# OLD — deprecated since LangGraph v1.0
from langgraph.prebuilt import create_react_agent
```

Running this produced a warning:

```
LangGraphDeprecatedSinceV10: create_react_agent has been moved to
`langchain.agents`. Please update your import to
`from langchain.agents import create_agent`.
```

### The Fix

```python
# NEW — current recommended import
from langchain.agents import create_agent
```

### Why the change?

| | `create_react_agent` (old) | `create_agent` (new) |
|---|---|---|
| Location | `langgraph.prebuilt` | `langchain.agents` |
| Name | Explicitly says "react" pattern | Generic — may support other patterns |
| API | `prompt` parameter | `system_prompt` parameter |
| Status | Deprecated (will be removed in v2.0) | Current |

> **Gotcha:** The parameter name changed too! The old function used `prompt=`, the new one uses `system_prompt=`. If you copy old tutorials, you'll get `TypeError: got an unexpected keyword argument 'prompt'`.

---

## 6. Step 3: String Model vs. ChatAnthropic Object

You can pass the model in two ways:

### Option A: String (simple)

```python
agent = create_agent(
    model="claude-sonnet-4-6",  # just a string
    tools=[get_weather],
)
```

### Option B: ChatAnthropic object (configurable)

```python
from langchain_anthropic import ChatAnthropic

agent = create_agent(
    model=ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=1024,
        timeout=30,
        max_retries=2,
    ),
    tools=[get_weather],
)
```

### When to use which?

| | String | ChatAnthropic |
|---|---|---|
| Simplicity | Minimal code | More boilerplate |
| Configuration | Default settings only | Full control |
| Temperature | Can't set | `temperature=0.7` |
| Token limits | Can't set | `max_tokens=1024` |
| Timeouts | Can't set | `timeout=30` |
| Retries | Can't set | `max_retries=2` |

**Rule of thumb:** Start with the string. Switch to `ChatAnthropic` when you need to tune behavior.

> **Gotcha: Model IDs** — If you're using a specific model version, you need the full ID: `claude-sonnet-4-20250514`, not `claude-sonnet-4-6`. The short alias (`claude-sonnet-4-6`) works as a string passed to `create_agent`, but may not work with `ChatAnthropic` depending on your API plan. Always check which model IDs your API key has access to.

### What about `init_chat_model`?

There's a third option — provider-agnostic:

```python
from langchain.chat_models import init_chat_model

# Switch providers by changing strings — no import changes
llm = init_chat_model("claude-sonnet-4-6", model_provider="anthropic")
llm = init_chat_model("gpt-4o", model_provider="openai")
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
```

Use `init_chat_model` if you might switch providers later. Use `ChatAnthropic` if you're committed to Claude.

---

## 7. Step 4: Preventing Infinite Loops

The agent's ReAct loop (reason → act → observe) could theoretically run forever. What if the agent keeps calling tools without reaching a conclusion?

### The Safety Net: `recursion_limit`

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={"recursion_limit": 10},
)
```

| Setting | What Happens |
|---|---|
| No limit set | Default is 25 steps (safe but generous) |
| `recursion_limit=10` | Stops after 10 steps, raises `GraphRecursionError` |
| `recursion_limit=3` | Very tight — might cut off legitimate multi-tool calls |

### Visualizing the loop:

```
Step 1: REASON  → "I need weather for Tokyo"
Step 2: ACT     → get_weather("Tokyo")
Step 3: OBSERVE → "27°C, Sunny"
Step 4: REASON  → "I have the answer"
Step 5: RESPOND → Final answer

Total: 5 steps (well within limit of 10)
```

For a simple weather agent, `recursion_limit=10` is plenty. For complex agents that chain many tools, you might need 20-30.

---

## 8. Step 5: Input Validation with Pydantic

Before the agent even sees the input, we validate it using Pydantic:

```python
from pydantic import BaseModel, Field, ValidationError


class UserQuery(BaseModel):
    city: str = Field(
        min_length=2,
        max_length=100,
        description="City name to get weather for"
    )
```

### Why Pydantic over dataclasses?

| | `dataclass` | `pydantic.BaseModel` |
|---|---|---|
| **Validation** | None — accepts anything | Auto-validates at runtime |
| **Coercion** | `age="25"` stays a string | `age="25"` becomes `int(25)` |
| **Error messages** | You write them | Auto-generated, human-readable |
| **JSON Schema** | No | Yes — used by LLMs for tool calling |

```python
# dataclass — silently accepts bad data
from dataclasses import dataclass

@dataclass
class UserDC:
    name: str
    age: int

u = UserDC(name=123, age="not a number")  # No error! Silently wrong.

# Pydantic — catches bad data immediately
from pydantic import BaseModel

class UserPD(BaseModel):
    name: str
    age: int

u = UserPD(name=123, age="25")    # name → "123", age → 25 (coerced)
u = UserPD(name="Ali", age=-1)    # ValidationError if you add ge=0
```

### Using it in our agent:

```python
raw_input = input("Enter a city name: ").strip()

try:
    user_query = UserQuery(city=raw_input)
except ValidationError as e:
    for error in e.errors():
        print(f"Invalid input: {error['msg']}")
    return
```

**Results:**
- Empty input → `"String should have at least 2 characters"`
- `"A"` → `"String should have at least 2 characters"`
- `"Chennai"` → Passes validation, proceeds to agent

---

## 9. Step 6: LLM-Powered City Validation

Pydantic validates *format* (length, type). But `"xyztown"` passes format validation — it's 7 characters. We need the LLM to validate *semantics* (is this a real city?).

The solution is beautifully simple — just tell the agent in the system prompt:

```python
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=(
        "You are a helpful assistant. "
        "Before using the get_weather tool, validate that the user's "
        "input is a real city in the world. "
        "If it is not a valid city name, politely tell the user "
        "and do not call the tool."
    ),
)
```

Now you have **two layers of validation**:

```
User Input: "xyztown"
     │
     ▼
┌─────────────────────┐
│ Layer 1: Pydantic    │  ← Format check (length, type)
│ Is it 2-100 chars?   │
│ Result: ✅ Pass       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Layer 2: LLM         │  ← Semantic check (is it real?)
│ Is "xyztown" a real  │
│ city in the world?   │
│ Result: ❌ Fail       │
└─────────┬───────────┘
          │
          ▼
Agent: "xyztown does not appear to be a real city.
        Could you please provide a valid city name?"
```

> **Why not just use the LLM for all validation?** Cost and speed. Pydantic catches obvious issues (empty strings, absurdly long input) for free in milliseconds. The LLM costs tokens and takes seconds. Use cheap validation first, expensive validation second.

---

## 10. Step 7: ToolRuntime — Giving Tools Context

This is where things get interesting.

### The Problem

Without `ToolRuntime`, your tool is blind — it only sees what the LLM explicitly passes:

```python
def get_weather(city: str) -> str:
    # I only know the city. Nothing else.
    # Who asked? No idea.
    # What was the conversation? No idea.
    # Do they prefer Celsius? No idea.
    return f"It's always sunny in {city}!"
```

### The Analogy

Imagine you work at a help desk:

- **Without ToolRuntime** = A customer calls. You only hear their question. You don't know who's calling, what they asked before, or their preferences.
- **With ToolRuntime** = You have a screen showing the caller's name, their call history, and saved preferences. You can give a personalized answer.

### The Solution

```python
from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import tool

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    # Who is asking?
    user_id = runtime.config.get("configurable", {}).get("user_id", "unknown")

    # What are their preferences?
    prefs = USER_PREFERENCES.get(user_id, {"unit": "celsius"})

    # What's the conversation history?
    message_count = len(runtime.state.get("messages", []))

    temp_c = 32
    if prefs["unit"] == "celsius":
        return f"Weather in {city}: {temp_c}°C, Sunny."
    else:
        temp_f = round(temp_c * 9 / 5 + 32)
        return f"Weather in {city}: {temp_f}°F, Sunny."
```

### The Key Insight

The LLM **never sees** the `runtime` parameter. It's hidden from the tool schema. The LLM just says "call get_weather with city=Chennai." LangGraph injects `runtime` automatically behind the scenes.

```
LLM sees this:                    Tool actually receives this:
┌──────────────────┐              ┌──────────────────────────┐
│ get_weather      │              │ get_weather              │
│                  │              │                          │
│ Args:            │              │ Args:                    │
│   city: str      │              │   city: str              │
│                  │              │   runtime: ToolRuntime   │
│ (that's all)     │              │     ├─ config            │
│                  │              │     ├─ state             │
└──────────────────┘              │     └─ store             │
                                  └──────────────────────────┘
```

### What ToolRuntime contains:

| Field | Purpose | Example |
|---|---|---|
| `config` | Run configuration | `{"configurable": {"user_id": "user_1"}}` |
| `state` | Current conversation state | `{"messages": [...]}` |
| `store` | Persistent key-value memory | Cross-session user data |
| `tool_call_id` | Unique ID of this tool call | For tracking/logging |
| `stream_writer` | For streaming partial results | Real-time updates |

### When do you need it?

| Scenario | Need ToolRuntime? |
|---|---|
| Simple tool, no user context | No |
| Tool needs to know which user is asking | Yes |
| Tool needs conversation history | Yes |
| Tool needs to save/load data across sessions | Yes |
| Different behavior per user | Yes |

### Passing user context at invocation:

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Weather in Tokyo?"}]},
    config={
        "recursion_limit": 10,
        "configurable": {"user_id": "user_1"},  # ← this reaches runtime.config
    },
)
```

---

## 11. Step 8: Configuring the LLM

With `ChatAnthropic`, you can fine-tune the model's behavior:

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.7,      # Creativity level
    max_tokens=1024,      # Response length cap
    timeout=30,           # Seconds before timeout
    max_retries=2,        # Retries on transient errors
)
```

### What each setting does:

**temperature** — Controls randomness in the response.

```
temperature=0.0  →  "The weather in Tokyo is 25°C and sunny."
temperature=0.0  →  "The weather in Tokyo is 25°C and sunny."  (identical)

temperature=1.0  →  "Tokyo's basking in glorious 25°C sunshine today!"
temperature=1.0  →  "It's a balmy 25 degrees in Tokyo with clear skies!"  (varied)
```

**max_tokens** — Caps the response length. Prevents the agent from writing essays when you want a short answer. 1024 tokens is roughly 750 words.

**timeout** — If the API doesn't respond in 30 seconds, fail fast instead of hanging. Critical for production apps.

**max_retries** — Network blips happen. This retries automatically on transient errors (500s, timeouts) without you writing retry logic.

---

## 12. Step 9: Multi-Turn Conversation with InMemorySaver

### The Problem

Without a checkpointer, every `agent.invoke()` is a clean slate:

```
You: What's the weather in Tokyo?
Bot: It's 25°C and sunny in Tokyo!

You: How about compared to London?
Bot: I don't know what you're comparing to.  ← No memory!
```

### The Solution

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=llm,
    tools=[get_weather],
    checkpointer=InMemorySaver(),  # ← enables conversation memory
)
```

### The Key: `thread_id`

The `thread_id` groups messages into a conversation:

```python
config = {
    "configurable": {
        "thread_id": "session_1",  # same thread = same conversation
        "user_id": "user_1",
    }
}

# These two calls share the same conversation memory
agent.invoke({"messages": [...]}, config=config)  # "Weather in Tokyo?"
agent.invoke({"messages": [...]}, config=config)  # "Compare to London?"
```

Different `thread_id` = different conversation. It's like opening a new chat window.

### Building the chat loop:

```python
print("Weather Agent (type 'quit' to exit)")

while True:
    raw_input = input("\nYou: ").strip()

    if raw_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    result = agent.invoke(
        {"messages": [{"role": "user", "content": raw_input}]},
        config=config,
    )

    last_message = result["messages"][-1]
    print(f"Agent: {last_message.content}")
```

### Result:

```
You: What's the weather in Tokyo?
Agent: Tokyo is 13.3°C and mainly clear.

You: How about Chennai?
Agent: Chennai is 27.3°C and partly cloudy.

You: Which one is hotter?
Agent: Chennai is hotter at 27.3°C compared to Tokyo's 13.3°C —
       about 14°C warmer!  ← Remembers both!
```

### InMemorySaver vs. other checkpointers:

| Checkpointer | Persistence | Best for |
|---|---|---|
| `InMemorySaver` | Lost when process stops | Development, testing |
| `SqliteSaver` | Saved to local file | Single-server apps |
| `PostgresSaver` | Saved to database | Production, multi-server |

---

## 13. Step 10: Real Weather Data with Open-Meteo

Our mock tool returns fake data. Let's connect to [Open-Meteo](https://open-meteo.com) — a free weather API that requires no API key.

### The architecture:

```
User: "Weather in Chennai?"
  │
  ▼
Agent calls: get_weather("Chennai")
  │
  ▼
┌─────────────────────────────────────────────┐
│ Step 1: Geocoding API                        │
│ "Chennai" → lat: 13.08, lon: 80.27          │
│                                              │
│ Step 2: Weather API                          │
│ lat/lon → temp: 27.3°C, humidity: 90%,      │
│           wind: 4.2 km/h, condition: Cloudy  │
└─────────────────────────────────────────────┘
  │
  ▼
Agent formats and responds to user
```

### The geocoding helper:

```python
import httpx


def _geocode(city: str) -> tuple[float, float, str]:
    """Convert city name to coordinates using Open-Meteo geocoding API."""
    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    )
    resp.raise_for_status()
    data = resp.json()
    if "results" not in data:
        raise ValueError(f"City '{city}' not found")
    result = data["results"][0]
    return result["latitude"], result["longitude"], result.get("name", city)
```

### The weather fetcher:

```python
def _fetch_weather(lat: float, lon: float, unit: str) -> dict:
    """Fetch current weather from Open-Meteo API."""
    temp_unit = "celsius" if unit == "celsius" else "fahrenheit"
    resp = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit": temp_unit,
            "wind_speed_unit": "kmh",
        },
    )
    resp.raise_for_status()
    return resp.json()["current"]
```

### WMO weather codes:

The API returns numeric codes for weather conditions. We map them to readable strings:

```python
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}
```

### The complete tool:

```python
@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get live weather for a given city using Open-Meteo API."""
    user_id = runtime.config.get("configurable", {}).get("user_id", "unknown")
    prefs = USER_PREFERENCES.get(user_id, {"unit": "celsius", "language": "en"})

    try:
        lat, lon, resolved_name = _geocode(city)
        weather = _fetch_weather(lat, lon, prefs["unit"])

        temp = weather["temperature_2m"]
        humidity = weather["relative_humidity_2m"]
        wind = weather["wind_speed_10m"]
        condition = WEATHER_CODES.get(weather["weather_code"], "Unknown")
        unit_symbol = "°C" if prefs["unit"] == "celsius" else "°F"

        return (
            f"Weather in {resolved_name}: {temp}{unit_symbol}, {condition}, "
            f"Humidity: {humidity}%, Wind: {wind} km/h"
        )
    except ValueError as e:
        return str(e)
    except httpx.HTTPError:
        return f"Failed to fetch weather data for {city}. Please try again."
```

Notice how `ToolRuntime` makes this work — the same tool returns Celsius for one user and Fahrenheit for another, without the LLM knowing about it.

---

## 14. The Final Code

Here's the complete working agent with all the concepts combined:

```python
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import httpx
from pydantic import BaseModel, Field, ValidationError
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver


# User preferences stored per user
USER_PREFERENCES = {
    "user_1": {"unit": "celsius", "language": "en"},
    "user_2": {"unit": "fahrenheit", "language": "es"},
}

# Weather code to description mapping (WMO codes)
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _geocode(city: str) -> tuple[float, float, str]:
    """Convert city name to coordinates using Open-Meteo geocoding API."""
    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    )
    resp.raise_for_status()
    data = resp.json()
    if "results" not in data:
        raise ValueError(f"City '{city}' not found")
    result = data["results"][0]
    return result["latitude"], result["longitude"], result.get("name", city)


def _fetch_weather(lat: float, lon: float, unit: str) -> dict:
    """Fetch current weather from Open-Meteo API."""
    temp_unit = "celsius" if unit == "celsius" else "fahrenheit"
    resp = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit": temp_unit,
            "wind_speed_unit": "kmh",
        },
    )
    resp.raise_for_status()
    return resp.json()["current"]


@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get live weather for a given city using Open-Meteo API."""
    user_id = runtime.config.get("configurable", {}).get("user_id", "unknown")
    prefs = USER_PREFERENCES.get(user_id, {"unit": "celsius", "language": "en"})

    try:
        lat, lon, resolved_name = _geocode(city)
        weather = _fetch_weather(lat, lon, prefs["unit"])

        temp = weather["temperature_2m"]
        humidity = weather["relative_humidity_2m"]
        wind = weather["wind_speed_10m"]
        condition = WEATHER_CODES.get(weather["weather_code"], "Unknown")
        unit_symbol = "°C" if prefs["unit"] == "celsius" else "°F"

        return (
            f"Weather in {resolved_name}: {temp}{unit_symbol}, {condition}, "
            f"Humidity: {humidity}%, Wind: {wind} km/h"
        )
    except ValueError as e:
        return str(e)
    except httpx.HTTPError:
        return f"Failed to fetch weather data for {city}. Please try again."


def main():
    agent = create_agent(
        model=ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=1024,
            timeout=30,
            max_retries=2,
        ),
        tools=[get_weather],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "You are a helpful weather assistant. "
            "Before using the get_weather tool, validate that the user's "
            "input is a real city in the world. "
            "If it is not a valid city name, politely tell the user "
            "and do not call the tool. "
            "You can reference previous messages in the conversation."
        ),
    )

    config = {
        "recursion_limit": 10,
        "configurable": {"thread_id": "session_1", "user_id": "user_1"},
    }

    print("Weather Agent (type 'quit' to exit)")
    print("=" * 50)

    while True:
        raw_input = input("\nYou: ").strip()

        if raw_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": raw_input}]},
            config=config,
        )

        last_message = result["messages"][-1]
        print(f"Agent: {last_message.content}")


if __name__ == "__main__":
    main()
```

### Run it:

```bash
uv run python main.py
```

### Sample session:

```
Weather Agent (type 'quit' to exit)
==================================================

You: What's the weather in Chennai?
Agent: The current weather in Chennai is:
  Temperature: 27.3°C
  Conditions: Partly cloudy
  Humidity: 90%
  Wind: 4.2 km/h

You: How about Tokyo?
Agent: The current weather in Tokyo is:
  Temperature: 13.3°C
  Conditions: Mainly clear
  Humidity: 94%
  Wind: 2.3 km/h

You: Which city is hotter?
Agent: Chennai is hotter at 27.3°C compared to Tokyo's 13.3°C —
       about 14°C warmer!

You: quit
Goodbye!
```

---

## 15. Key Concepts Recap

Here's everything we covered, mapped to the code:

```
┌────────────────────────────────────────────────────────┐
│                    OUR AGENT                            │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ChatAnthropic (Step 3, 8)                        │   │
│  │ The "brain" — configured with temperature,       │   │
│  │ max_tokens, timeout, max_retries                  │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐   │
│  │ create_agent (Step 2)                             │   │
│  │ Builds the ReAct loop with system_prompt          │   │
│  │ + tools + checkpointer + response_format          │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐   │
│  │ Tools with ToolRuntime (Step 7)                   │   │
│  │ Functions the agent can call, with access to      │   │
│  │ user context, state, and config                   │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐   │
│  │ InMemorySaver (Step 9)                            │   │
│  │ Conversation memory via thread_id                 │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐   │
│  │ Pydantic + System Prompt (Step 5, 6)              │   │
│  │ Two-layer validation: format + semantics          │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐   │
│  │ recursion_limit (Step 4)                          │   │
│  │ Safety net against infinite loops                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

---

## 16. Where Agentic AI Is Used Today

The pattern we built — chat interface, tools, memory — is the foundation of real products:

### Chat Interface
**Examples:** ChatGPT, Claude, customer support bots (Intercom, Zendesk)

The user types natural language, the agent reasons and calls tools to respond.

### Copilot / Side Panel
**Examples:** GitHub Copilot, Microsoft 365 Copilot, Cursor

An agent embedded alongside your main application, offering suggestions and taking actions within the app.

### Voice Assistant
**Examples:** Alexa, Siri, Google Assistant

Same ReAct pattern, but with speech-to-text input and text-to-speech output.

### Autonomous Task Runner
**Examples:** Devin (coding), Replit Agent, Claude Code

The agent runs multi-step tasks with minimal human input — planning, executing, and verifying on its own.

### Workflow Automation
**Examples:** Zapier AI, n8n AI nodes

No chat UI at all — the agent triggers on events (new email, new ticket) and processes them automatically.

### How our agent maps to production:

| Our Agent (Learning) | Production App |
|---|---|
| Terminal `input()` | Web UI (React, Svelte) or mobile app |
| `InMemorySaver` | `PostgresSaver` for persistent sessions |
| Single tool | Multiple tools (weather, news, maps, calendar) |
| `USER_PREFERENCES` dict | Database with real user profiles |
| `print()` output | Streaming API via FastAPI / WebSocket |

---

## 17. Common Gotchas

Here's every issue we hit during development, so you don't have to:

### 1. Deprecated imports
```python
# WRONG — will be removed in LangGraph v2.0
from langgraph.prebuilt import create_react_agent

# RIGHT
from langchain.agents import create_agent
```

### 2. Parameter name changed
```python
# WRONG — old API
agent = create_agent(model=llm, prompt="You are helpful")

# RIGHT — new API
agent = create_agent(model=llm, system_prompt="You are helpful")
```

### 3. Model ID not found (404 error)
```python
# May fail if your API plan doesn't have access
ChatAnthropic(model="claude-sonnet-4-6-20250514")

# Use the model ID your plan supports
ChatAnthropic(model="claude-sonnet-4-20250514")
```

Always check which models your API key has access to at [console.anthropic.com](https://console.anthropic.com).

### 4. Windows Unicode encoding error
```
UnicodeEncodeError: 'charmap' codec can't encode characters
```

The LLM returns emojis and special characters that Windows PowerShell can't display. Fix:

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
```

### 5. API key not set in new terminal
```
TypeError: Could not resolve authentication method.
Expected either api_key or auth_token to be set.
```

Each new terminal session needs the API key. Set it permanently:

```powershell
# Windows — persist across all sessions
[System.Environment]::SetEnvironmentVariable(
    "ANTHROPIC_API_KEY", "your-key", "User"
)
```

### 6. LLMs don't have real-time data
The LLM cannot tell you today's weather, stock prices, or news. It only knows its training data. For real-time information, you **must** provide tools that call external APIs.

### 7. response_format vs. free-form in multi-turn chat
If you use `response_format=WeatherResponse`, *every* response must fit that schema. This breaks when the agent needs to respond with comparisons, follow-ups, or error messages. For interactive chat, free-form text is more flexible.

### 8. @tool decorator
When using `ToolRuntime`, you need the `@tool` decorator from `langchain_core.tools`. Without it, `ToolRuntime` injection won't work — it's how LangGraph knows to intercept the parameter.

```python
# WRONG — ToolRuntime won't be injected
def get_weather(city: str, runtime: ToolRuntime) -> str:
    ...

# RIGHT
@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    ...
```

---

## What's Next?

From here, you could:

- **Add more tools** — time zones, currency, news alongside weather
- **Add streaming** — show responses token by token
- **Persist memory** — replace `InMemorySaver` with `SqliteSaver` or `PostgresSaver`
- **Build a web UI** — FastAPI backend + Svelte frontend
- **Go multi-agent** — a supervisor that routes to specialized sub-agents

The pattern stays the same: define tools, create an agent, invoke with context. The complexity comes from what your tools do, not from the agent framework itself.

---

*Built with LangChain, Claude Sonnet 4, and Open-Meteo. All code is available and ready to run.*

*If this helped you understand agentic AI, give it a clap and follow for more hands-on AI engineering guides!*

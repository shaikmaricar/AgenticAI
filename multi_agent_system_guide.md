# Multi-Agent AI System — A Complete Walkthrough

Building a team of cooperating AI agents with **LangGraph**, **Tavily**, and **OpenRouter**.

---

## 1. What Are We Building?

This project builds a small "team" of three specialised AI agents that work together to answer a user's question. Instead of one giant LLM call trying to do everything, we split the work:

| Agent | Role | What it does |
|---|---|---|
| **Supervisor** | The manager | Decides who works next based on the current state |
| **Research Agent** | The researcher | Searches the web using Tavily for up-to-date facts |
| **Computation Agent** | The thinker | Uses an LLM to reason over the data and produce a final answer |

The whole system is wired together as a **state graph** using LangGraph. Think of it as a flowchart where each box is an agent (a Python function) and the arrows are decided dynamically based on what's currently in the shared "state."

### Why This Pattern?

A single LLM call is fine for trivia like "What's the capital of France?", but real questions often need two distinct skills:

1. **Fetching external information** — e.g. "What is the latest population of Japan?" The LLM doesn't know this; it needs to look it up.
2. **Reasoning on top of that data** — e.g. "...and double it." That's arithmetic on the result of step 1.

By giving each capability its own agent and letting a supervisor coordinate them, we get a system that is more reliable, easier to debug, and easier to extend (want to add a "code execution agent"? Just add a node).

A key design choice here: **the supervisor uses plain Python `if`/`else` logic, not an LLM.** This is deliberate. LLM-based routers can hallucinate, loop forever, or burn tokens making routing decisions. Deterministic routing is free, fast, and impossible to confuse.

---

## 2. The Imports — What Each Library Does

```python
from dotenv import load_dotenv
import os
from typing import Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_community.tools.tavily_search import TavilySearchResults
```

### `dotenv` and `os`

`python-dotenv` reads a `.env` file in your project folder and loads its `KEY=value` pairs into environment variables. We use this so API keys live in a file you can `.gitignore` rather than being hard-coded in source. The `os` module is Python's built-in interface to environment variables — `os.getenv("KEY")` reads them, `os.environ["KEY"] = ...` writes them.

### `Literal` and `TypedDict`

These two are about *type hints*, but they're not just for documentation — LangGraph actually inspects them at runtime.

`Literal["a", "b", "c"]` says "this value can only be one of these exact strings." We use it to declare which nodes a function is allowed to route to. LangGraph reads these annotations to draw the graph's edges, so getting them right is part of the contract.

`TypedDict` defines a dictionary with a fixed set of keys and a known type for each. At runtime it behaves like a regular `dict`, but type checkers and LangGraph use the schema to validate state updates. We import from `typing_extensions` (not `typing`) for the most up-to-date version that works across Python versions.

### `ChatOpenAI`

LangChain's wrapper for any OpenAI-compatible chat API. The name is misleading — despite being called `ChatOpenAI`, it works with **any** provider that speaks the OpenAI API format: OpenRouter, Together, Groq, local Ollama, vLLM, and so on. You just point it at a different `base_url`.

### `@tool` decorator

Turns a regular Python function into a "tool" that an agent can call. It auto-extracts the function's name, docstring, and argument types to build a tool schema the LLM understands. Decorating with `@tool` is the difference between "a function I can call from Python" and "a capability the agent system knows about."

### `StateGraph`, `END`, and `Command`

These three together are the heart of LangGraph. `StateGraph` is the directed graph: nodes are functions that read and update a shared state, and edges describe how control flows. `END` is a special sentinel node meaning "we're done — return the final state." `Command` is a return value that does two things at once: tells LangGraph **where to go next** (`goto=...`) and **what to update** in the shared state (`update=...`). Returning a `Command` keeps routing logic *inside* the node, which is much cleaner than configuring it externally.

### `TavilySearchResults`

Tavily is a search API built specifically for LLM agents. Unlike scraping Google, it returns clean, summarised text designed to be dropped into prompts. `TavilySearchResults` is the LangChain wrapper around its REST API. (In newer versions of `langchain_community` this is being superseded by `langchain_tavily.TavilySearch`, but the older import still works.)

---

## 3. Environment Setup

```python
load_dotenv()
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
```

`load_dotenv()` reads your `.env` file and pushes the variables into `os.environ`. After this call, anything in `.env` is accessible via `os.getenv("KEY_NAME")`.

The next two lines look redundant — we're reading and writing the same variables — but it's a belt-and-braces pattern. Some libraries check `os.environ` directly rather than going through `os.getenv` (which is just a safer wrapper). Explicitly assigning to `os.environ` guarantees the keys are there no matter how the downstream library looks them up.

Your `.env` file should look like this:

```
OPENROUTER_API_KEY=sk-or-...
TAVILY_API_KEY=tvly-...
```

---

## 4. Configuring the LLM

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
```

Each parameter matters:

**`model="gpt-4o-mini"`** — the model identifier. Because we're routing through OpenRouter, you can swap this for any model OpenRouter supports: `mistralai/mistral-7b-instruct`, `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.1-70b-instruct`, etc. If you got a "model not found" error, try the prefixed form `openai/gpt-4o-mini` — OpenRouter sometimes requires the provider prefix.

**`api_key=os.getenv(...)`** — auth token, pulled from environment so it never appears in source.

**`base_url="https://openrouter.ai/api/v1"`** — this is the magic line. It redirects `ChatOpenAI` away from OpenAI's servers and over to OpenRouter's. Because OpenRouter speaks the OpenAI API format, the same client class works unchanged. Remove this line if you want to talk to OpenAI directly.

**`temperature=0`** — controls randomness. At `0`, the model picks the highest-probability token at each step, giving deterministic, repeatable output. Higher values (`0.7`, `1.0`) introduce variety, which is good for creative writing but bad for factual or arithmetic tasks. Since our system answers factual questions, `0` is the right choice.

---

## 5. The Shared State

```python
class AgentState(TypedDict):
    user_query: str
    research_data: str
    final_answer: str
    next_agent: str
```

Every node in the graph reads from and writes to this single shared dictionary. Think of it as the team's whiteboard — everyone can see it and update it.

`user_query` is the original question, set once at startup and treated as read-only after. `research_data` holds raw text from the web search, starts empty, and gets filled by the research agent. `final_answer` holds the polished answer from the computation agent. `next_agent` is a breadcrumb showing which agent the previous node *thinks* should run next; it's not strictly required (the supervisor makes its own decisions), but it's invaluable when you `print(state)` to debug a trace.

Defining the shape with `TypedDict` gives autocomplete in your editor, type-checker validation, and a single source of truth for what flows through the system.

---

## 6. Defining the Tool

```python
tavily_tool = TavilySearchResults(max_results=3)

@tool
def web_search(query: str):
    """Search the web using Tavily."""
    results = tavily_tool.invoke(query)
    return str(results)
```

`TavilySearchResults(max_results=3)` is the underlying search client. The `max_results=3` parameter is a deliberate trade-off: fewer results means less noise, less prompt context consumed, and cheaper LLM calls; more results means better recall but more tokens and more cost. Three is a good default for factual lookups.

Wrapping `tavily_tool` in our own `@tool`-decorated function is a common pattern even when the wrapped tool is already a LangChain tool. It gives us a single place to control the docstring (which the LLM sees as the tool's description), massage the output format (here, forcing it to a string so it slots into prompts cleanly), and add things like caching, logging, or retries later without touching the rest of the code.

`tavily_tool.invoke(query)` is the standard LangChain method — every `Runnable` (LLMs, tools, chains, graphs) exposes `.invoke()`. It returns a list of dicts with `title`, `content`, and `url` for each result. We `str()` it so it can drop straight into a prompt.

---

## 7. The Supervisor Agent

```python
def supervisor_agent(
    state: AgentState,
) -> Command[Literal["research_agent", "computation_agent", END]]:
    """Deterministic supervisor logic to prevent infinite loops."""

    if state.get("final_answer"):
        return Command(goto=END)

    if not state.get("research_data"):
        return Command(goto="research_agent")

    if state.get("research_data") and not state.get("final_answer"):
        return Command(goto="computation_agent")

    return Command(goto=END)
```

The return type annotation — `Command[Literal["research_agent", "computation_agent", END]]` — tells LangGraph "this function returns a `Command` whose `goto` will be one of these three values." LangGraph uses that to draw the graph's edges automatically, so you don't have to declare them by hand.

The logic itself reads top-to-bottom as a priority list:

**Rule 1:** if a `final_answer` is already set, we're done — go to `END`. `state.get("key")` returns `None` (falsy) if the key is missing or its value is an empty string, so this only triggers when there's a real answer.

**Rule 2:** if no research has been done yet (empty `research_data`), send work to the research agent first. This is what happens on the very first tick.

**Rule 3:** if research is done but no answer has been computed, hand off to the computation agent.

**Fallback:** the final `return Command(goto=END)` should be unreachable given the rules above, but having a default exit is good defensive programming — the graph can never accidentally hang.

The whole function runs in microseconds and costs nothing. That's the point.

---

## 8. The Research Agent

```python
def research_agent(state: AgentState) -> Command[Literal["supervisor_agent"]]:
    """Uses Tavily to gather external information."""
    search_result = web_search.invoke(state["user_query"])
    return Command(
        goto="supervisor_agent",
        update={
            "research_data": search_result,
            "next_agent": "supervisor_agent",
        },
    )
```

Single responsibility: take the user's question, search the web, stash the results into state, hand control back to the supervisor.

The return type `Command[Literal["supervisor_agent"]]` declares "after I run, the next stop is always the supervisor." That's a static edge, not a dynamic decision.

`state["user_query"]` uses dictionary bracket access — fine here because `user_query` is always present (it's set by the caller). If you wanted defensive code for an optional key, you'd use `state.get("user_query")`.

The `update` dict in the returned `Command` is **merged** into the shared state by LangGraph. We're saying: "when you go to the supervisor, also write `research_data` and `next_agent` into the state first." This is how state mutates across nodes.

---

## 9. The Computation Agent

```python
def computation_agent(state: AgentState) -> Command[Literal["supervisor_agent"]]:
    """Uses reasoning based on research data or user query."""

    prompt = f"""
You are a reasoning agent.

User Query:
{state['user_query']}

Research Data (if any):
{state.get('research_data', '')}

Provide a clear final answer.
"""
    response = llm.invoke(prompt)
    return Command(
        goto="supervisor_agent",
        update={
            "final_answer": response.content,
            "next_agent": "supervisor_agent",
        },
    )
```

This is where the actual reasoning happens.

The prompt is built with an **f-string** (the `f"""..."""` syntax). Anything inside `{curly_braces}` is evaluated as Python and inserted into the string. So `{state['user_query']}` becomes the actual question, and `{state.get('research_data', '')}` becomes the search results.

Notice the asymmetry in how we access each field. We use `state['user_query']` (bracket access) because the query is always present — if it weren't, that's a bug we want to crash on. We use `state.get('research_data', '')` (with a default) because in edge cases the supervisor might route here without research being done; the empty-string fallback prevents a `KeyError`.

`llm.invoke(prompt)` sends the text to the LLM and returns an `AIMessage` object. The actual generated text lives in `response.content` — that's why we extract it before storing.

After the LLM responds, control flows back to the supervisor, which on its next tick will see `final_answer` is now set and route to `END`.

---

## 10. Building the Graph

```python
graph = StateGraph(AgentState)

graph.add_node("supervisor_agent", supervisor_agent)
graph.add_node("research_agent", research_agent)
graph.add_node("computation_agent", computation_agent)

graph.set_entry_point("supervisor_agent")

graph.add_edge("research_agent", "supervisor_agent")
graph.add_edge("computation_agent", "supervisor_agent")

app = graph.compile()
```

`StateGraph(AgentState)` creates an empty graph and tells it which `TypedDict` defines the state's shape. LangGraph uses this schema to validate every update returned from a node.

`graph.add_node("name", function)` registers a node. The string name is what other parts of the graph use to refer to it (in `goto=...`, in edges); the function is what actually executes when the node is visited. The string and the variable name don't have to match, but keeping them identical makes debugging easier.

`graph.set_entry_point("supervisor_agent")` is where execution begins when you call `app.invoke()`. The supervisor inspects the initial state and routes from there.

`graph.add_edge(...)` declares **static** edges — these always fire, no decision needed. We don't need to manually wire edges out of the supervisor because the supervisor returns `Command(goto=...)`, which routes itself. But the worker agents always loop back to the supervisor, so we declare those as fixed edges. The supervisor then re-evaluates state and decides what's next.

`graph.compile()` validates everything (no orphan nodes, no edges to nowhere) and returns a `Runnable` you can call with `.invoke()`, `.stream()`, etc.

---

## 11. Running It

```python
if __name__ == "__main__":
    result = app.invoke({
        "user_query": "What is the latest population of Japan and double it?",
        "research_data": "",
        "final_answer": "",
        "next_agent": "",
    })
    print(result["final_answer"])
```

The `if __name__ == "__main__":` guard is standard Python — it means "only run this block when the file is executed directly, not when it's imported as a module." Useful when you later want to import these agents into another script.

The example query is intentionally chosen to need *both* agents: "latest population of Japan" requires web research, "double it" requires arithmetic reasoning on top of that result. Either capability alone wouldn't suffice.

We initialise all four state keys, even though only `user_query` carries real data. This avoids `None` versus `""` ambiguity later and matches the `TypedDict` schema cleanly.

`app.invoke(...)` runs the graph to completion and returns the final state — a dict with all four keys populated. We pull out just the `final_answer` to print.

---

## 12. The Execution Trace

For the example query, here's exactly what happens:

```
1. invoke()
   state = {user_query: "What is the latest population of Japan and double it?",
            research_data: "", final_answer: "", next_agent: ""}

2. supervisor_agent runs
   -> sees research_data is empty
   -> Command(goto="research_agent")

3. research_agent runs
   -> calls Tavily, gets search results
   -> Command(goto="supervisor_agent",
              update={research_data: "<search results>", ...})

4. supervisor_agent runs (second time)
   -> sees research_data is set, final_answer still empty
   -> Command(goto="computation_agent")

5. computation_agent runs
   -> builds prompt with query + research data
   -> calls LLM
   -> Command(goto="supervisor_agent",
              update={final_answer: "About 247 million...", ...})

6. supervisor_agent runs (third time)
   -> sees final_answer is set
   -> Command(goto=END)

7. invoke() returns the final state. Done.
```

Three trips through the supervisor, each one a deterministic `if`/`else` decision. No loops, no hallucinated routing, no mystery.

---

## 13. Where to Go From Here

Once you have this working, the natural extensions are:

**More agents.** Add a `code_agent` that runs Python via a sandbox, or a `summary_agent` that compresses long research data before reasoning. Each is just another node and another supervisor rule.

**Streaming.** Replace `app.invoke(...)` with `app.stream(...)` to get incremental updates as each node runs — great for showing progress in a UI.

**Persistence.** Pass a `checkpointer` to `graph.compile()` so state survives across runs. This is how you build agents that remember previous conversations.

**Better routing.** If your rules get complex, you can swap the deterministic supervisor for an LLM-based one — but keep this version around as a fallback. Deterministic routing is almost always the right starting point.

The pattern itself — supervisor + specialised workers + shared state — scales surprisingly far before you need anything fancier.

"""
================================================================================
  MULTI-AGENT AI SYSTEM (LangGraph + Tavily + OpenRouter)
================================================================================

OVERVIEW
--------
This script builds a small "team" of AI agents that cooperate to answer a user
question. The team has three roles:

  1. SUPERVISOR AGENT   - the "manager". Decides who works next.
  2. RESEARCH AGENT     - the "researcher". Searches the web (via Tavily).
  3. COMPUTATION AGENT  - the "thinker". Uses an LLM to reason over the data
                          and produce a final answer.

The whole thing is wired together as a STATE GRAPH using LangGraph.
Think of it as a flowchart where each box is an agent (a Python function),
and the arrows are decided dynamically based on the current "state".

WHY THIS PATTERN?
-----------------
A single LLM call is great for simple Q&A, but real questions often need:
  - external/up-to-date information (e.g. "latest population of Japan")
  - then some reasoning/computation on top (e.g. "...and double it")

By splitting the work into specialised agents and letting a supervisor route
between them, we get something more reliable and debuggable than one giant
prompt. The supervisor logic here is *deterministic* (plain if/else), which
prevents the classic "agent loops forever" failure mode.
================================================================================
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
# Each import below is explained in detail so you understand WHY it's needed,
# not just what it is.

# --- Standard library ---------------------------------------------------------

from dotenv import load_dotenv
# `python-dotenv` reads a `.env` file in your project folder and loads the
# key=value pairs into environment variables. We use this so we don't have to
# hard-code API keys (which would be a security risk if you push to GitHub).
# Your `.env` file should look like:
#       OPENROUTER_API_KEY=sk-or-...
#       TAVILY_API_KEY=tvly-...

import os
# `os` is Python's built-in module for talking to the operating system.
# We use `os.getenv("KEY")` to read environment variables, and
# `os.environ["KEY"] = ...` to set them for libraries that look them up.

# --- Typing helpers -----------------------------------------------------------

from typing import Literal
# `Literal` lets us say "this value can ONLY be one of these exact strings."
# Example: Literal["research_agent", "computation_agent", END]
# This isn't just for documentation - LangGraph actually inspects these type
# hints to figure out which nodes a function is allowed to route to. So getting
# the Literal right is important; it's part of the graph's contract.

from typing_extensions import TypedDict
# `TypedDict` lets us define a dictionary with a fixed set of keys and a known
# type for each key. It behaves like a normal dict at runtime, but type
# checkers (and LangGraph) can validate the shape.
# We import from `typing_extensions` (not `typing`) because it has the most
# up-to-date version that works across Python versions.

# --- LangChain: the LLM wrapper -----------------------------------------------

from langchain_openai import ChatOpenAI
# `ChatOpenAI` is LangChain's wrapper for any OpenAI-compatible chat API.
# Despite the name, it's NOT limited to OpenAI - any provider that speaks the
# OpenAI API format works (OpenRouter, Together, Groq, local Ollama, etc.).
# We'll point it at OpenRouter below.

from langchain_core.tools import tool
# `@tool` is a decorator that turns a regular Python function into a "tool"
# that LangChain/LangGraph agents can call. It auto-extracts the function's
# name, docstring, and argument types to build a tool schema.

# --- LangGraph: the agent orchestration framework -----------------------------

from langgraph.graph import StateGraph, END
# `StateGraph` is the core building block. It's a directed graph where:
#   - nodes are functions (agents) that read and update a shared state
#   - edges describe how control flows from one node to the next
# `END` is a special sentinel node that means "we're done, return the state."

from langgraph.types import Command
# `Command` is a return value that tells LangGraph two things at once:
#   1. WHERE to go next (the `goto` parameter)
#   2. WHAT to update in the shared state (the `update` parameter)
# Returning a Command is more flexible than `add_conditional_edges`, because
# the routing decision lives inside the node itself.

# --- LangChain Community: third-party tool integrations -----------------------

from langchain_community.tools.tavily_search import TavilySearchResults
# Tavily is a search API built specifically for LLM agents - it returns
# clean, summarised results instead of raw HTML. `TavilySearchResults` is
# the LangChain wrapper around the Tavily REST API.
# Note: in newer versions this may be replaced by `langchain_tavily.TavilySearch`.


# =============================================================================
# SECTION 2: ENVIRONMENT SETUP
# =============================================================================
# Loading API keys from `.env` so we never hard-code secrets in source files.

load_dotenv()
# Reads the `.env` file in the current directory (or any parent directory)
# and pushes those variables into `os.environ`. After this call, anything
# stored in `.env` is available via `os.getenv("KEY_NAME")`.

os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
# These two lines look redundant (we're reading and writing the same vars),
# but they serve a purpose: they GUARANTEE the keys are in `os.environ` even
# if some library only checks `os.environ` directly (rather than `os.getenv`,
# which is just a safer wrapper around it). A bit of a belt-and-braces step.


# =============================================================================
# SECTION 3: LLM CONFIGURATION
# =============================================================================
# Setting up the language model that the computation_agent will use to think.

llm = ChatOpenAI(
    model="gpt-4o-mini",
    #   The model identifier. Because we're routing through OpenRouter,
    #   you can swap this for any model OpenRouter supports, e.g.
    #   "mistralai/mistral-7b-instruct", "anthropic/claude-3.5-sonnet", etc.
    #   For pure OpenAI you'd use names like "gpt-4o" or "gpt-4o-mini".
    #
    #   Tip: if you get a "model not found" error on OpenRouter, try the
    #   prefixed form like "openai/gpt-4o-mini" - OpenRouter sometimes
    #   requires the provider prefix.

    api_key=os.getenv("OPENROUTER_API_KEY"),
    #   The auth token. Pulled from environment so it's never in source code.

    base_url="https://openrouter.ai/api/v1",
    #   This is the magic line that redirects ChatOpenAI away from OpenAI's
    #   servers and over to OpenRouter's. OpenRouter speaks the OpenAI API
    #   format, so the same client class works unchanged.
    #   Remove this line entirely if you want to use OpenAI directly.

    temperature=0,
    #   Controls randomness in generation:
    #     0   = deterministic, picks the highest-probability token every time.
    #           Good for facts, math, code - anywhere you want consistency.
    #     0.7 = balanced creativity (a common default for chatbots).
    #     1.0+ = highly creative / varied output.
    #   We use 0 here because we want repeatable, factual answers.
)


# =============================================================================
# SECTION 4: SHARED STATE DEFINITION
# =============================================================================
# Every node in the graph reads from and writes to a single shared dictionary
# called the "state". Defining its shape with TypedDict gives us autocomplete,
# type checking, and a single source of truth for what data flows through.

class AgentState(TypedDict):
    user_query: str
    #   The original question the user asked. Set once at startup, then
    #   read-only for the rest of the run.

    research_data: str
    #   Raw text from the web search. Empty string until research_agent runs.
    #   The supervisor uses this field to decide whether research is "done".

    final_answer: str
    #   The polished answer produced by computation_agent. Empty string until
    #   that agent runs. The supervisor uses this to decide if we can stop.

    next_agent: str
    #   A breadcrumb showing which agent the previous node thinks should run
    #   next. This isn't strictly required (the supervisor makes its own
    #   decisions) but it's useful for logging and debugging the trace.


# =============================================================================
# SECTION 5: TOOL DEFINITION
# =============================================================================
# Tools are external capabilities that an agent can invoke. Here we define
# one tool: web search via Tavily.

tavily_tool = TavilySearchResults(max_results=3)
#   `TavilySearchResults` is the underlying search client.
#   `max_results=3` limits how many results come back per query. Why 3?
#     - Fewer = less noise, less prompt context used, cheaper LLM calls.
#     - More  = better recall but more tokens / more cost.
#   Three is a good default balance for factual lookups.


@tool
def web_search(query: str):
    """Search the web using Tavily."""
    # Wrapping `tavily_tool` in our own `@tool`-decorated function is a common
    # pattern. It gives us a place to:
    #   - control the docstring (which becomes the tool description the LLM
    #     sees when deciding whether to call it)
    #   - massage the output (here, force it to a string)
    #   - add caching, logging, retries, etc. later

    results = tavily_tool.invoke(query)
    # `.invoke(...)` is the standard LangChain method for running any
    # Runnable component (LLMs, tools, chains - all use `.invoke`).
    # It returns a list of dicts with title/content/url for each result.

    return str(results)
    # We stringify so the result can be dropped straight into a prompt or
    # stored in the state without any further processing.


# =============================================================================
# SECTION 6: THE SUPERVISOR AGENT
# =============================================================================
# The supervisor is the brain of the operation. It looks at the current state
# and decides which worker agent should run next. Crucially, it uses *plain
# Python logic* rather than asking an LLM - this makes routing decisions:
#   - free (no API calls)
#   - fast (microseconds)
#   - deterministic (no surprise loops)

def supervisor_agent(
    state: AgentState,
) -> Command[Literal["research_agent", "computation_agent", END]]:
    """Deterministic supervisor logic to prevent infinite loops."""
    # The return type annotation tells LangGraph: "this function will return
    # a Command whose `goto` is one of these three values." LangGraph uses
    # this to draw the edges of the graph automatically.

    # ---- Rule 1: If we already have a final answer, we're done. ----
    if state.get("final_answer"):
        return Command(goto=END)
    # `state.get("key")` returns the value if present, else None (which is
    # falsy). So this triggers when `final_answer` is a non-empty string.

    # ---- Rule 2: If no research has been done yet, do that first. ----
    if not state.get("research_data"):
        return Command(goto="research_agent")
    # First time through, `research_data` is "" (falsy), so we go research.

    # ---- Rule 3: We have research but no answer -> go reason about it. ----
    if state.get("research_data") and not state.get("final_answer"):
        return Command(goto="computation_agent")
    # By now `research_data` is populated. We hand off to the thinker.

    # ---- Fallback safety net ----
    return Command(goto=END)
    # Should be unreachable given the rules above, but it's good practice
    # to have a default exit so the graph can't accidentally hang.


# =============================================================================
# SECTION 7: THE RESEARCH AGENT
# =============================================================================
# This agent's only job: take the user's question, search the web, and stash
# the results into the state. Then it hands control back to the supervisor.

def research_agent(state: AgentState) -> Command[Literal["supervisor_agent"]]:
    """Uses Tavily to gather external information."""
    # Return type says "after I run, the next stop is supervisor_agent."

    search_result = web_search.invoke(state["user_query"])
    # Calls our `@tool`-decorated function with the original question.
    # `state["user_query"]` is the bracket-style access that TypedDict
    # supports. (We could also use `state.get("user_query")` for safety.)

    return Command(
        goto="supervisor_agent",
        # After researching, return to the supervisor for routing.

        update={
            "research_data": search_result,
            # Persist the search results into shared state so the
            # computation_agent can read them on the next hop.

            "next_agent": "supervisor_agent",
            # Just a breadcrumb - useful when you `print(state)` for debugging.
        },
    )


# =============================================================================
# SECTION 8: THE COMPUTATION AGENT
# =============================================================================
# This is where the actual "thinking" happens. It builds a prompt that
# includes the user's question + any research data, sends it to the LLM,
# and stores the response as the final answer.

def computation_agent(state: AgentState) -> Command[Literal["supervisor_agent"]]:
    """Uses reasoning based on research data or user query."""

    # f-string templating: anything inside {curly_braces} is evaluated as a
    # Python expression and inserted into the string.
    prompt = f"""
You are a reasoning agent.

User Query:
{state['user_query']}

Research Data (if any):
{state.get('research_data', '')}

Provide a clear final answer.
"""
    # Note the use of `state.get('research_data', '')`:
    #   - the second argument is a default returned if the key is missing
    #   - this guards against KeyError if `research_data` was never set
    #   - useful for queries that don't need research (the supervisor
    #     would still route here as a fallback)

    response = llm.invoke(prompt)
    # `.invoke(prompt)` sends the text to the LLM and returns an
    # `AIMessage` object. The actual text is in `response.content`.

    return Command(
        goto="supervisor_agent",
        # Hand back to the supervisor, which will see `final_answer` is now
        # set and route to END on the next tick.

        update={
            "final_answer": response.content,
            # `.content` extracts the plain text answer from the AIMessage.

            "next_agent": "supervisor_agent",
        },
    )


# =============================================================================
# SECTION 9: BUILDING THE GRAPH
# =============================================================================
# Now we wire the agents together into an executable workflow.

graph = StateGraph(AgentState)
# Create an empty graph and tell it which TypedDict defines the state's shape.
# LangGraph uses this to validate updates returned from nodes.

# ---- Register each node ----
# Signature: graph.add_node("name_used_in_routing", function_to_run)
# The string name is how other parts of the graph refer to this node;
# the function is what actually runs when the node is visited.

graph.add_node("supervisor_agent", supervisor_agent)
graph.add_node("research_agent", research_agent)
graph.add_node("computation_agent", computation_agent)

# ---- Set the starting node ----
graph.set_entry_point("supervisor_agent")
# When `app.invoke(...)` runs, execution begins here. The supervisor will
# inspect the initial state and decide where to go first.

# ---- Define the static edges ----
# We don't need to manually wire edges OUT of the supervisor, because the
# supervisor returns `Command(goto=...)` which dynamically routes itself.
# But research and computation always loop back to the supervisor, so we
# add those as fixed edges:

graph.add_edge("research_agent", "supervisor_agent")
graph.add_edge("computation_agent", "supervisor_agent")
# Reads as: "after research_agent finishes, always go to supervisor_agent."
# The supervisor then re-evaluates the state and decides what's next.

# ---- Compile the graph into an executable app ----
app = graph.compile()
# `compile()` validates the graph (no orphans, valid edges, etc.) and
# returns a Runnable you can call with `.invoke()`, `.stream()`, etc.


# =============================================================================
# SECTION 10: RUNNING THE SYSTEM
# =============================================================================
# Demo: ask a question that needs both web research and arithmetic reasoning.

if __name__ == "__main__":
    # The `if __name__ == "__main__":` guard is standard Python boilerplate.
    # It means "only run this block if the file was executed directly,
    # not if it was imported as a module from another file."

    result = app.invoke({
        "user_query": "What is the latest population of Japan and double it?",
        # The actual question. Notice it has two parts:
        #   - a fact lookup ("latest population of Japan")  -> research_agent
        #   - a computation ("double it")                    -> computation_agent
        # That's why this multi-agent design helps.

        "research_data": "",   # Starts empty; research_agent will fill it.
        "final_answer": "",    # Starts empty; computation_agent will fill it.
        "next_agent": "",      # Starts empty; updated as the graph runs.
    })

    # `result` is the final state dict, with all four keys populated.
    # We pull out just the final answer for display.
    print(result["final_answer"])


# =============================================================================
# EXECUTION TRACE (for the example query)
# =============================================================================
#  1. invoke()  -> state = {user_query: "...", research_data: "", final_answer: "", ...}
#  2. supervisor: research_data is empty  -> goto research_agent
#  3. research_agent: calls Tavily, fills research_data  -> goto supervisor
#  4. supervisor: research_data set, final_answer empty  -> goto computation_agent
#  5. computation_agent: LLM reasons + answers, fills final_answer -> goto supervisor
#  6. supervisor: final_answer set  -> goto END
#  7. invoke() returns the final state. Done!
# =============================================================================

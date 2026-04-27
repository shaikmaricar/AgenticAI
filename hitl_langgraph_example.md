# Human-in-the-Loop (HITL) with LangGraph — Annotated Walkthrough

This notebook builds a small calculator agent that pauses for human approval whenever it detects a **risky** operation (e.g., dividing by zero, or an unrecognized command). It's a clean illustration of the HITL pattern: an automated pipeline with a human gate inserted at the right moment.

---

## 1. Imports & Environment Setup

```python
from dotenv import load_dotenv
import os
from typing import TypedDict, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
```

**Why these imports?**
- `dotenv` + `os` → load secrets (API keys) from a `.env` file instead of hardcoding them.
- `TypedDict`, `Literal` → give the shared graph state a typed schema so each node knows the contract.
- `ChatOpenAI` → the LLM client (used here via OpenRouter, which exposes many models behind an OpenAI-compatible API).
- `StateGraph`, `END` → the building blocks of any LangGraph: a directed graph of nodes that pass a shared state, with `END` marking termination.
- `Command`, `interrupt` → LangGraph's first-class HITL primitives. `interrupt` pauses execution; `Command` resumes it with new state.

```python
load_dotenv()
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
```

Loads the `.env` file and explicitly puts the key into the process environment so downstream libraries can pick it up.

---

## 2. LLM Configuration

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",          # You can also use "mistralai/mistral-7b-instruct"
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)
```

**Why this configuration?**
- `base_url` points at OpenRouter, not OpenAI directly — this lets you swap models with one string change.
- `temperature=0` makes the model deterministic, which is what you want for a parser/analyzer that should behave the same way every time.
- (Note: in this particular example the LLM is configured but the parsing is done with regex — the LLM is set up so you can extend the agent later.)

---

## 3. Shared State Definition

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
import re

# Define Shared State
class TaskState(TypedDict):
    user_input: str
    operation: str
    numbers: list
    risk_level: str
    human_decision: str
    result: str
    final_output: str
```

**Why a TypedDict?**
Every node in a LangGraph receives the same `state` object and returns a partial dict that gets merged into it. `TypedDict` documents exactly which keys exist and what types they hold, so:
- IDE autocomplete works.
- Mistakes like `state["operaton"]` get caught early.
- New collaborators can see the data flow at a glance.

Each field maps to a stage of the pipeline:
| Field | Set by | Purpose |
|---|---|---|
| `user_input` | caller | Raw natural-language request |
| `operation` | parse_intent | One of `add/subtract/multiply/divide/unknown` |
| `numbers` | parse_intent | Extracted integers |
| `risk_level` | risk_analysis | `HIGH` or `LOW` |
| `human_decision` | human_review | `approve` / `modify` / `reject` |
| `result` | execute_operation | Numeric result as string |
| `final_output` | format_response | User-facing message |

---

## 4. Node 1 — Intent Parser

```python
# Intent Parser Node
# Extract operation + numbers
def parse_intent(state: TaskState):
    text = state["user_input"].lower()

    numbers = list(map(int, re.findall(r'\d+', text)))

    if "add" in text:
        operation = "add"
    elif "subtract" in text:
        operation = "subtract"
    elif "multiply" in text:
        operation = "multiply"
    elif "divide" in text:
        operation = "divide"
    else:
        operation = "unknown"

    print(f"\nParsed Operation: {operation}")
    print(f"Extracted Numbers: {numbers}")

    return {
        "operation": operation,
        "numbers": numbers
    }
```

**What it does:** Lowercases the input, pulls out every integer with a regex, and matches the verb against a small whitelist.

**Why this way?** It's a deliberately simple keyword parser to keep the focus on the HITL pattern. In production you'd let the LLM extract structured intent — but the rest of the graph would be identical.

**Why return a partial dict?** LangGraph nodes return *only the keys they want to update*. The framework merges them into the existing state, so we don't have to copy unchanged fields.

---

## 5. Node 2 — Risk Analyzer

```python
# Risk Analyzer Node
def risk_analysis(state: TaskState):
    if state["operation"] == "divide" and 0 in state["numbers"]:
        risk = "HIGH"
    elif state["operation"] == "unknown":
        risk = "HIGH"
    else:
        risk = "LOW"

    print(f"Risk Level: {risk}")
    return {"risk_level": risk}
```

**What it does:** Flags two conditions as risky:
1. Division involving a `0` (would crash with `ZeroDivisionError`).
2. An unknown operation (we have no idea what to do).

**Why a separate node?** This is the *gate* — it decides whether the request is safe to auto-execute or needs a human's eyes. Keeping it isolated means you can later swap the rule-based check for an LLM-based safety classifier without touching the rest of the graph.

---

## 6. Node 3 — Human Approval (the HITL step)

```python
# Human Approval Node
def human_review(state: TaskState):

    print("\n--- HUMAN REVIEW REQUIRED ---")
    print("Operation:", state["operation"])
    print("Numbers:", state["numbers"])

    decision = input(
        "Type 'approve' to continue, "
        "'modify' to change operation, "
        "or 'reject': "
    ).lower()

    if decision == "modify":
        new_op = input("Enter new operation (add/subtract/multiply/divide): ")
        return {
            "human_decision": "approved",
            "operation": new_op.lower()
        }

    return {"human_decision": decision}
```

**What it does:** Prints the proposed operation, asks the human what to do, and writes their answer back into the state.

**Three possible decisions:**
- `approve` → continue with the original operation.
- `modify` → human supplies a corrected operation; we mark it `approved` so the executor still runs, but with the new op.
- `reject` → execution will be skipped by the formatter.

**Why `input()` here?** This is a notebook-friendly demo. In a real deployment you would use LangGraph's `interrupt()` (imported above), which serializes state, pauses the graph, and lets a UI/API resume it later via a `Command(resume=...)`. The logic is the same; only the I/O changes.

---

## 7. Node 4 — Operation Executor

```python
# Operation Executor
def execute_operation(state: TaskState):

    op = state["operation"]
    nums = state["numbers"]

    try:
        if op == "add":
            result = sum(nums)

        elif op == "subtract":
            result = nums[0] - nums[1]

        elif op == "multiply":
            result = nums[0] * nums[1]

        elif op == "divide":
            result = nums[0] / nums[1]

        else:
            return {"result": "Invalid operation"}

        return {"result": str(result)}

    except Exception as e:
        return {"result": f"Execution Error: {str(e)}"}
```

**What it does:** Performs the arithmetic. Wrapped in `try/except` so a bad input (e.g., division by zero that somehow got past the gate) is caught and reported as a result string instead of crashing the graph.

**Why convert the result to `str`?** The `TaskState` declares `result: str`. Keeping a single type avoids surprises downstream when the formatter does string interpolation.

---

## 8. Node 5 — Response Formatter

```python
# Response Formatter Node
def format_response(state: TaskState):

    if state.get("human_decision") == "reject":
        return {"final_output": "Operation Rejected by Human"}

    return {
        "final_output": f"Operation '{state['operation']}' "
                        f"executed successfully. Result = {state['result']}"
    }
```

**What it does:** Produces the final user-facing message. If the human rejected the operation, it short-circuits with a rejection notice; otherwise it confirms the result.

**Why `state.get(...)` instead of `state[...]`?** `human_decision` only exists when the risk path went through review. `.get()` returns `None` for the low-risk path that skipped human approval, avoiding a `KeyError`.

---

## 9. Routing Logic (conditional edge)

```python
# Routing Logic
def route_after_risk(state: TaskState):
    if state["risk_level"] == "HIGH":
        return "human_review"
    return "execute_operation"
```

**What it does:** This isn't a node — it's a *router* used by `add_conditional_edges`. It returns the **name** of the next node based on the current state. High-risk → human gate; low-risk → straight to execution.

---

## 10. Wiring the Graph

```python
graph = StateGraph(TaskState)

graph.add_node("parse_intent", parse_intent)
graph.add_node("risk_analysis", risk_analysis)
graph.add_node("human_review", human_review)
graph.add_node("execute_operation", execute_operation)
graph.add_node("format_response", format_response)

graph.set_entry_point("parse_intent")

graph.add_edge("parse_intent", "risk_analysis")

graph.add_conditional_edges(
    "risk_analysis",
    route_after_risk,
    {
        "human_review": "human_review",
        "execute_operation": "execute_operation"
    }
)

graph.add_edge("human_review", "execute_operation")
graph.add_edge("execute_operation", "format_response")
graph.add_edge("format_response", END)

app = graph.compile()
```

**The flow it builds:**

```
parse_intent → risk_analysis → ┬── (LOW)  → execute_operation → format_response → END
                               └── (HIGH) → human_review ──────┘
```

**Key points:**
- `set_entry_point` defines where execution starts.
- `add_edge` is an *unconditional* transition.
- `add_conditional_edges` takes a router function and a mapping from its return values to node names. This is where the HITL branching happens.
- Both branches reconverge at `execute_operation`, so the rest of the pipeline is shared.
- `graph.compile()` turns the definition into a runnable `app`.

---

## 11. Running the Example

```python
# Run Example
initial_state = {
    "user_input": "Divide 100 by 0",
    "operation": "",
    "numbers": [],
    "risk_level": "",
    "human_decision": "",
    "result": "",
    "final_output": ""
}

result = app.invoke(initial_state)

print("\nFinal Output:")
print(result["final_output"])
```

**Why pre-fill all keys with empty values?** `TypedDict` doesn't enforce required keys at runtime, but pre-filling makes the state shape explicit and avoids `KeyError`s if any node reads a field before it's been set.

---

## 12. Sample Run Trace

With `"Divide 100 by 0"` as input:

```
Parsed Operation: divide
Extracted Numbers: [100, 0]
Risk Level: HIGH

--- HUMAN REVIEW REQUIRED ---
Operation: divide
Numbers: [100, 0]
> modify
> add

Final Output:
Operation 'add' executed successfully. Result = 100
```

The human caught the divide-by-zero, switched the operation to `add`, and the executor produced `100 + 0 = 100`. That's the whole point of HITL: the automation does the easy 95%, and a person handles the edge cases the system flagged.

---

---

## 13. Production Pattern — `interrupt()` + `Command`

`input()` works perfectly for learning and notebooks, but it ties the human step to the Python process that's running the graph. The moment you want a web UI, a Slack approval button, or any pause that survives a server restart, you need LangGraph's native HITL primitives: `interrupt()` and `Command`.

### How it differs from `input()`

| Concern | `input()` version | `interrupt()` version |
|---|---|---|
| Where the human responds | Same terminal/notebook | Any UI — web, Slack, email, API |
| What happens during the pause | Process blocks | Graph state is *checkpointed*, process is free |
| Survives a restart? | No | Yes (with a persistent checkpointer) |
| Multiple users / async approvals | No | Yes |
| Required setup | Nothing | A checkpointer + a `thread_id` |

### The `human_review` node, rewritten

```python
from langgraph.types import interrupt

def human_review(state: TaskState):
    print("\n--- HUMAN REVIEW REQUIRED ---")

    # interrupt() pauses the graph and surfaces this payload to the caller.
    # Whatever the caller passes to Command(resume=...) becomes the return value.
    decision_payload = interrupt({
        "question": "Approve, modify, or reject?",
        "operation": state["operation"],
        "numbers": state["numbers"],
    })

    # decision_payload is whatever the UI/API sent back — typically a dict.
    decision = decision_payload.get("decision", "").lower()

    if decision == "modify":
        return {
            "human_decision": "approved",
            "operation": decision_payload.get("new_operation", "").lower()
        }

    return {"human_decision": decision}
```

**What changed:**
- `input()` is replaced by `interrupt(payload)`. The payload is whatever metadata the UI needs to render the approval screen.
- The function "returns" via `interrupt()` *and resumes from that same line* when the graph is invoked again with a `Command`. That's the magic — LangGraph re-runs the node and `interrupt()` now returns the resumed value instead of pausing.

### Compiling with a checkpointer

`interrupt()` only works if the graph has somewhere to save state during the pause:

```python
from langgraph.checkpoint.memory import MemorySaver
# For real persistence: from langgraph.checkpoint.sqlite import SqliteSaver
#                      from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

- `MemorySaver` → in-process, lost on restart. Fine for testing.
- `SqliteSaver` / `PostgresSaver` → durable. Use these in production so a server crash doesn't lose pending approvals.

### Running it (two phases)

```python
from langgraph.types import Command

# A thread_id lets LangGraph correlate the pause with the resume.
config = {"configurable": {"thread_id": "task-001"}}

# Phase 1: kick off the graph. It runs until it hits interrupt(),
# then returns control with the interrupt payload visible in the result.
result = app.invoke(initial_state, config=config)

# At this point your UI shows the approval screen.
# The graph state is safely checkpointed.

# Phase 2: human responds — could be seconds or days later, from any process
# that can talk to the same checkpointer.
result = app.invoke(
    Command(resume={"decision": "modify", "new_operation": "add"}),
    config=config,
)

print(result["final_output"])
```

The two `invoke` calls share a `thread_id`, which is how LangGraph knows the second call is a resume of the first.

### When to migrate

Stay on `input()` while you're:
- Learning the graph structure
- Iterating on node logic in a notebook
- Building a personal CLI tool

Switch to `interrupt()` as soon as you:
- Add any kind of UI (web, mobile, chat)
- Need approvals to survive a restart
- Have approvers who aren't the same person running the script
- Want to handle multiple concurrent tasks

The graph topology, state schema, and every other node stay identical — only the human node and the invocation pattern change. That's why it's a small refactor *now* and a much bigger one once your codebase has grown around `input()`.

---

## Key Takeaways

1. **Separate the gate from the work.** A dedicated risk-analysis node makes the HITL trigger easy to reason about and easy to upgrade later (rules → ML model → LLM).
2. **Conditional edges are the HITL hinge.** They let the same graph behave fully autonomously *or* hand off to a human depending on state.
3. **State is the contract.** Once you've nailed down `TaskState`, every node is independent and testable.
4. **`input()` is a demo stand-in for `interrupt()`.** For production, swap to LangGraph's interrupt/Command pattern so the pause survives across processes and UIs.

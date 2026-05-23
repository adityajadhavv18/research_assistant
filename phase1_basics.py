import os 
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END

load_dotenv()

# ── 1. DEFINE STATE ──────────────────────────────────────────────────────────
# State is just a TypedDict — a dictionary with typed keys.
# Every node in the graph gets the whole state, and returns an update to it.

class ResearchState(TypedDict):
    query: str
    plan: str
    results: list
    report: str
    step: int

# ── 2. DEFINE NODES ──────────────────────────────────────────────────────────
# A node is just a Python function that:
#   - receives the current state (dict)
#   - does some work
#   - returns a dict of keys to UPDATE in the state

def greet_node(state: ResearchState) -> dict:
    """First node: just prints the query so we can see state flowing through."""
    print(f"\n[greet_node] Received query: '{state['query']}'")
    return {"step": 1}   # only update what changed

def echo_node(state: ResearchState) -> dict:
    """Second node: echoes back a dummy result so we can see chaining."""
    print(f"[echo_node]  Step number is now: {state['step']}")
    return {"report": f"Placeholder report for: {state['query']}", "step": 2}


def done_node(state: ResearchState) -> dict:
    """Final node: prints the finished state."""
    print(f"[done_node]  Report ready: '{state['report']}'")
    return {}   # nothing to update, just a terminal step


# ── 3. BUILD THE GRAPH ───────────────────────────────────────────────────────
# StateGraph wires nodes together using the State schema.


builder = StateGraph(ResearchState)


# Add nodes — first arg is a name (string), second is the function
builder.add_node("greet", greet_node)
builder.add_node("echo", echo_node)
builder.add_node("done", done_node)

# Add edges — tells LangGraph which node runs after which

builder.set_entry_point("greet")
builder.add_edge("greet","echo")
builder.add_edge("echo","done")
builder.add_edge("done", END)


graph = builder.compile()

if __name__ == "__main__":
    initial_state = {
        "query": "What are the latest breakthroughs in fusion energy?",
        "plan": "",
        "results": [],
        "report": "",
        "step": 0,
    }

    print("=== Running Phase 1 Graph ===")
    final_state = graph.invoke(initial_state)

    print("\n=== Final State ===")
    for key, value in final_state.items():
        print(f"  {key}: {value}")
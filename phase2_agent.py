# phase2_agent.py
# Phase 2: Real LLM calls + web search tool + research loop

import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

load_dotenv()

# ── CLIENTS ──────────────────────────────────────────────────────────────────
llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ── STATE ────────────────────────────────────────────────────────────────────
# Same shape as Phase 1 — we just fill more keys now

class ResearchState(TypedDict):
    query:        str    # original user question
    sub_questions: list  # planner breaks query into these
    results:      list   # raw search results accumulate here
    report:       str    # final synthesised answer
    loop_count:   int    # safety counter — stops infinite loops


# ── NODE 1: PLANNER ──────────────────────────────────────────────────────────
# Takes the user's big question and asks the LLM to break it into
# 2-3 focused sub-questions. More focused = better search results.

def planner_node(state: ResearchState) -> dict:
    print(f"\n[planner] Breaking down: '{state['query']}'")

    prompt = f"""You are a research planner.
Break this research question into exactly 3 focused sub-questions
that together would fully answer it.

Research question: {state['query']}

Return ONLY a numbered list, nothing else. Example:
1. Sub-question one
2. Sub-question two
3. Sub-question three"""

    response = llm.invoke(prompt)

    # Parse the numbered list into a Python list
    lines = response.content.strip().split("\n")
    sub_questions = [
        line.split(". ", 1)[1].strip()   # remove "1. " prefix
        for line in lines
        if line.strip() and line[0].isdigit()
    ]

    print(f"[planner] Sub-questions: {sub_questions}")
    return {
        "sub_questions": sub_questions,
        "loop_count": 0,
        "results": []        # reset results for fresh run
    }


# ── NODE 2: RESEARCHER ───────────────────────────────────────────────────────
# Each call searches for the NEXT unanswered sub-question.
# It picks which sub-question to search based on loop_count
# (first call searches sub_question[0], second searches [1], etc.)

def researcher_node(state: ResearchState) -> dict:
    idx = state["loop_count"]                   # which sub-question turn it is
    sub_q = state["sub_questions"][idx]         # grab the right sub-question

    print(f"\n[researcher] Searching for: '{sub_q}'")

    # Call Tavily — returns top 3 web results
    search_results = tavily.search(
        query=sub_q,
        max_results=3
    )

    # Extract just the useful bits from each result
    new_results = [
        {
            "sub_question": sub_q,
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "content": r.get("content", "")[:500]  # cap at 500 chars
        }
        for r in search_results.get("results", [])
    ]

    print(f"[researcher] Got {len(new_results)} results")

    # Append to existing results (don't overwrite — we accumulate)
    return {
        "results":    state["results"] + new_results,
        "loop_count": state["loop_count"] + 1
    }


# ── CONDITIONAL EDGE: SHOULD WE KEEP RESEARCHING? ───────────────────────────
# This is the router function for the conditional edge.
# It returns a string — LangGraph uses that string to pick the next node.
# Return "research" → loop back to researcher_node
# Return "write"    → move on to writer_node

def should_continue(state: ResearchState) -> str:
    searched_count = state["loop_count"]
    total_questions = len(state["sub_questions"])

    if searched_count < total_questions:
        print(f"\n[router] {searched_count}/{total_questions} done — keep researching")
        return "research"   # loop back
    else:
        print(f"\n[router] All {total_questions} sub-questions searched — writing report")
        return "write"      # move to writer


# ── NODE 3: WRITER ───────────────────────────────────────────────────────────
# Takes ALL accumulated results and asks the LLM to synthesise
# them into one coherent research report.

def writer_node(state: ResearchState) -> dict:
    print(f"\n[writer] Synthesising {len(state['results'])} results...")

    # Format results for the prompt
    results_text = ""
    for i, r in enumerate(state["results"], 1):
        results_text += f"\n[{i}] {r['title']}\n{r['content']}\n"

    prompt = f"""You are a research writer.
Using the search results below, write a clear, well-structured
research summary that answers this question:

QUESTION: {state['query']}

SEARCH RESULTS:
{results_text}

Write a 3-4 paragraph summary. Be factual, cite sources by number [1], [2] etc."""

    response = llm.invoke(prompt)
    print("[writer] Report written.")

    return {"report": response.content}


# ── BUILD THE GRAPH ──────────────────────────────────────────────────────────

builder = StateGraph(ResearchState)

# Add nodes
builder.add_node("planner",    planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer",     writer_node)

# Normal edges
builder.set_entry_point("planner")
builder.add_edge("planner", "researcher")     # always go to researcher after planning

# Conditional edge — THIS is the loop
# After researcher runs, call should_continue() to decide what's next
builder.add_conditional_edges(
    "researcher",           # source node
    should_continue,        # router function
    {
        "research": "researcher",   # "research" → loop back to researcher
        "write":    "writer",       # "write"    → go to writer
    }
)

builder.add_edge("writer", END)

graph = builder.compile()


# ── RUN IT ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initial_state = {
        "query":         "Who is Virat Kohli and what are his records?",
        "sub_questions": [],
        "results":       [],
        "report":        "",
        "loop_count":    0,
    }

    print("=== Phase 2: Research Agent ===")
    final_state = graph.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    print(final_state["report"])
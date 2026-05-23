# phase3_writer.py
# Upgrades: smarter conditional edge + structured writer + quality check node

import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

load_dotenv()

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ── STATE ─────────────────────────────────────────────────────────────────────
# Two new keys added: quality_score and quality_feedback

class ResearchState(TypedDict):
    query:          str
    sub_questions:  list
    results:        list
    report:         str
    loop_count:     int
    quality_score:  int    # NEW — LLM scores the report 1 to 10
    quality_feedback: str  # NEW — LLM explains the score in one sentence


# ── NODE 1: PLANNER (unchanged) ───────────────────────────────────────────────

def planner_node(state: ResearchState) -> dict:
    print(f"\n[planner] Breaking down: '{state['query']}'")

    prompt = f"""You are a research planner.
Break this research question into exactly 3 focused sub-questions
that together would fully answer it.

Research question: {state['query']}

Return ONLY a numbered list, nothing else."""

    response = llm.invoke(prompt)
    lines = response.content.strip().split("\n")
    sub_questions = [
        line.split(". ", 1)[1].strip()
        for line in lines
        if line.strip() and line[0].isdigit()
    ]

    print(f"[planner] Sub-questions: {sub_questions}")
    return {
        "sub_questions":    sub_questions,
        "loop_count":       0,
        "results":          [],
        "quality_score":    0,
        "quality_feedback": ""
    }


# ── NODE 2: RESEARCHER (unchanged) ────────────────────────────────────────────

def researcher_node(state: ResearchState) -> dict:
    idx   = state["loop_count"]
    sub_q = state["sub_questions"][idx]

    print(f"\n[researcher] Searching: '{sub_q}'")

    search_results = tavily.search(query=sub_q, max_results=3)
    new_results = [
        {
            "sub_question": sub_q,
            "title":        r.get("title", ""),
            "url":          r.get("url", ""),
            "content":      r.get("content", "")[:500]
        }
        for r in search_results.get("results", [])
    ]

    print(f"[researcher] Got {len(new_results)} results")
    return {
        "results":    state["results"] + new_results,
        "loop_count": state["loop_count"] + 1
    }


# ── UPGRADED CONDITIONAL EDGE ─────────────────────────────────────────────────
# Phase 2: just counted loops
# Phase 3: LLM reads the results and JUDGES if they're good enough

def should_continue(state: ResearchState) -> str:

    # Safety cap — never loop more than 6 times no matter what
    if state["loop_count"] >= 6:
        print("\n[router] Hit safety limit — forcing write")
        return "write"

    # Still have unanswered sub-questions → keep searching
    if state["loop_count"] < len(state["sub_questions"]):
        print(f"\n[router] {state['loop_count']}/{len(state['sub_questions'])} done — keep researching")
        return "research"

    # All sub-questions searched → ask the LLM if results are good enough
    results_preview = "\n".join([r["title"] for r in state["results"]])

    prompt = f"""You are a research quality checker.

Original question: {state['query']}

Search results collected so far (titles only):
{results_preview}

Are these results sufficient to write a thorough answer?
Reply with ONLY one word: YES or NO."""

    response = llm.invoke(prompt)
    answer = response.content.strip().upper()

    print(f"\n[router] LLM quality check: {answer}")

    if "YES" in answer:
        return "write"
    else:
        # Not enough — add one more targeted search by appending a new sub-question
        print("[router] Results not sufficient — adding a follow-up search")
        state["sub_questions"].append(f"Latest developments in: {state['query']}")
        return "research"


# ── UPGRADED NODE 3: WRITER ───────────────────────────────────────────────────
# Phase 2: plain paragraph summary
# Phase 3: structured report with clear sections

def writer_node(state: ResearchState) -> dict:
    print(f"\n[writer] Synthesising {len(state['results'])} results...")

    results_text = ""
    for i, r in enumerate(state["results"], 1):
        results_text += f"\n[{i}] {r['title']}\nURL: {r['url']}\n{r['content']}\n"

    prompt = f"""You are a professional research writer.
Using the search results below, write a well-structured research report
that thoroughly answers the question.

QUESTION: {state['query']}

FORMAT YOUR REPORT EXACTLY LIKE THIS:
## Summary
(2-3 sentence overview of the key answer)

## Key Findings
(3-5 bullet points of the most important facts)

## Detailed Analysis
(2-3 paragraphs going deeper into the topic)

## Sources
(list each source as: [number] Title — URL)

SEARCH RESULTS:
{results_text}"""

    response = llm.invoke(prompt)
    print("[writer] Report written.")
    return {"report": response.content}


# ── NEW NODE 4: QUALITY CHECK ─────────────────────────────────────────────────
# Brand new in Phase 3.
# After the writer finishes, this node reads the report and scores it.
# The score and feedback get saved to state so Gradio can display them.

def quality_check_node(state: ResearchState) -> dict:
    print("\n[quality] Scoring the report...")

    prompt = f"""You are a research editor.
Read this research report and score its quality.

ORIGINAL QUESTION: {state['query']}

REPORT:
{state['report']}

Give:
1. A score from 1 to 10 (10 = excellent)
2. One sentence of feedback

Reply in EXACTLY this format and nothing else:
SCORE: 8
FEEDBACK: The report covers the main points well but lacks specific dates."""

    response = llm.invoke(prompt)
    lines = response.content.strip().split("\n")

    # Parse score and feedback from the LLM's response
    score    = 0
    feedback = ""
    for line in lines:
        if line.startswith("SCORE:"):
            try:
                score = int(line.split(":")[1].strip())
            except:
                score = 0
        if line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    print(f"[quality] Score: {score}/10 — {feedback}")
    return {
        "quality_score":    score,
        "quality_feedback": feedback
    }


# ── BUILD THE GRAPH ────────────────────────────────────────────────────────────

builder = StateGraph(ResearchState)

builder.add_node("planner",       planner_node)
builder.add_node("researcher",    researcher_node)
builder.add_node("writer",        writer_node)
builder.add_node("quality_check", quality_check_node)   # NEW

builder.set_entry_point("planner")
builder.add_edge("planner", "researcher")

builder.add_conditional_edges(
    "researcher",
    should_continue,
    {
        "research": "researcher",
        "write":    "writer",
    }
)

builder.add_edge("writer",        "quality_check")   # writer → quality check
builder.add_edge("quality_check", END)               # quality check → done

graph = builder.compile()


# ── RUN IT ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initial_state = {
        "query":            "Who is virat kohli?",
        "sub_questions":    [],
        "results":          [],
        "report":           "",
        "loop_count":       0,
        "quality_score":    0,
        "quality_feedback": "",
    }

    print("=== Phase 3: Research Agent ===")
    final_state = graph.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    print(final_state["report"])

    print("\n" + "="*50)
    print(f"QUALITY SCORE: {final_state['quality_score']}/10")
    print(f"FEEDBACK: {final_state['quality_feedback']}")
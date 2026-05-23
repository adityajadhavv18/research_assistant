# app.py  — Phase 4: Gradio UI
# Run with: python app.py

import os
import gradio as gr
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

load_dotenv()

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ── STATE ──────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query:            str
    sub_questions:    list
    results:          list
    report:           str
    loop_count:       int
    quality_score:    int
    quality_feedback: str


# ── NODES (same as Phase 3 — copy/paste) ──────────────────────────────────────

def planner_node(state: ResearchState) -> dict:
    prompt = f"""You are a research planner.
Break this research question into exactly 3 focused sub-questions.

Research question: {state['query']}

Return ONLY a numbered list, nothing else."""
    response = llm.invoke(prompt)
    lines = response.content.strip().split("\n")
    sub_questions = [
        line.split(". ", 1)[1].strip()
        for line in lines
        if line.strip() and line[0].isdigit()
    ]
    return {"sub_questions": sub_questions, "loop_count": 0, "results": [], "quality_score": 0, "quality_feedback": ""}


def researcher_node(state: ResearchState) -> dict:
    idx   = state["loop_count"]
    sub_q = state["sub_questions"][idx]
    search_results = tavily.search(query=sub_q, max_results=3)
    new_results = [
        {"sub_question": sub_q, "title": r.get("title",""), "url": r.get("url",""), "content": r.get("content","")[:500]}
        for r in search_results.get("results", [])
    ]
    return {"results": state["results"] + new_results, "loop_count": state["loop_count"] + 1}


def should_continue(state: ResearchState) -> str:
    if state["loop_count"] >= 6:
        return "write"
    if state["loop_count"] < len(state["sub_questions"]):
        return "research"
    results_preview = "\n".join([r["title"] for r in state["results"]])
    prompt = f"""Original question: {state['query']}
Search result titles: {results_preview}
Are these sufficient to write a thorough answer? Reply ONLY: YES or NO."""
    answer = llm.invoke(prompt).content.strip().upper()
    if "YES" in answer:
        return "write"
    state["sub_questions"].append(f"Latest developments in: {state['query']}")
    return "research"


def writer_node(state: ResearchState) -> dict:
    results_text = ""
    for i, r in enumerate(state["results"], 1):
        results_text += f"\n[{i}] {r['title']}\nURL: {r['url']}\n{r['content']}\n"
    prompt = f"""You are a professional research writer.
Write a structured report answering: {state['query']}

FORMAT EXACTLY:
## Summary
(2-3 sentence overview)

## Key Findings
(3-5 bullet points)

## Detailed Analysis
(2-3 paragraphs)

## Sources
([number] Title — URL)

SEARCH RESULTS:
{results_text}"""
    response = llm.invoke(prompt)
    return {"report": response.content}


def quality_check_node(state: ResearchState) -> dict:
    prompt = f"""Rate this research report.
QUESTION: {state['query']}
REPORT: {state['report']}

Reply EXACTLY:
SCORE: (1-10)
FEEDBACK: (one sentence)"""
    response = llm.invoke(prompt)
    lines    = response.content.strip().split("\n")
    score    = 0
    feedback = ""
    for line in lines:
        if line.startswith("SCORE:"):
            try: score = int(line.split(":")[1].strip())
            except: score = 0
        if line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()
    return {"quality_score": score, "quality_feedback": feedback}


# ── BUILD GRAPH ────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("planner",       planner_node)
    builder.add_node("researcher",    researcher_node)
    builder.add_node("writer",        writer_node)
    builder.add_node("quality_check", quality_check_node)
    builder.set_entry_point("planner")
    builder.add_edge("planner", "researcher")
    builder.add_conditional_edges("researcher", should_continue, {"research": "researcher", "write": "writer"})
    builder.add_edge("writer",        "quality_check")
    builder.add_edge("quality_check", END)
    return builder.compile()

graph = build_graph()


# ── GRADIO FUNCTION ────────────────────────────────────────────────────────────
# This is the key new concept in Phase 4.
# Instead of graph.invoke() which blocks until done,
# we use graph.stream() which yields updates after EACH node.
# We yield status updates to Gradio as each node completes —
# that's what makes the UI update live while the agent runs.

def run_research(query: str):
    yield ("Planning your research...", "*Starting...*", "", "")

    initial_state = {
        "query":            query,
        "sub_questions":    [],
        "results":          [],
        "report":           "",
        "loop_count":       0,
        "quality_score":    0,
        "quality_feedback": "",
    }

    final_state  = dict(initial_state)   # ← start with a copy
    search_count = 0
    import time
    start_time   = time.time()

    for update in graph.stream(initial_state):
        node_name = list(update.keys())[0]
        state     = update[node_name]

        final_state.update(state)        # ← merge, don't replace

        if node_name == "planner":
            sub_qs = final_state.get("sub_questions", [])
            status = "Planning done. Sub-questions:\n"
            for i, q in enumerate(sub_qs, 1):
                status += f"  {i}. {q}\n"
            status += "\nStarting research..."
            yield (status, "*Researching...*", "", "")

        elif node_name == "researcher":
            search_count += 1
            total  = len(final_state.get("sub_questions", []))
            result_count = len(final_state.get("results", []))
            status = f"Searching... ({search_count}/{total} done)\n"
            status += f"Results collected so far: {result_count}"
            yield (status, "*Researching...*", "", "")

        elif node_name == "writer":
            yield (
                "Writing complete. Running quality check...",
                final_state.get("report", ""),   # ← show report immediately
                "",
                ""
            )

        elif node_name == "quality_check":
            elapsed      = round(time.time() - start_time)
            score        = final_state.get("quality_score", 0)
            feedback     = final_state.get("quality_feedback", "")
            result_count = len(final_state.get("results", []))

            score_color = "🟢" if score >= 8 else ("🟡" if score >= 6 else "🔴")
            score_text  = f"{score_color} {score}/10\n{feedback}"
            stats_text  = (
                f"Searches: {search_count}\n"
                f"Results collected: {result_count}\n"
                f"LLM calls: 4\n"
                f"Time taken: {elapsed}s"
            )

            yield (
                "Research complete!",
                final_state.get("report", ""),
                score_text,
                stats_text
            )


# ── GRADIO UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Research Assistant") as demo:

    gr.Markdown("# Research Assistant")
    gr.Markdown("Powered by LangGraph + GPT-4o-mini + Tavily")

    # Input row
    with gr.Row():
        query_box  = gr.Textbox(
            placeholder="Ask a research question...",
            label="Your question",
            scale=4
        )
        submit_btn = gr.Button("Research", variant="primary", scale=1)

    # Status box — updates live as nodes complete
    status_box = gr.Textbox(
        label="Agent status",
        lines=5,
        interactive=False
    )

    # Report — appears once writer node finishes
    report_box = gr.Markdown(value="*Report will appear here...*")

    # Bottom row — score and stats side by side
    with gr.Row():
        score_box = gr.Textbox(label="Quality score", lines=3, interactive=False)
        stats_box = gr.Textbox(label="Run stats",     lines=3, interactive=False)

    # Wire the button to the function
    # outputs must match the 4 values we yield in run_research()
    submit_btn.click(
        fn=run_research,
        inputs=query_box,
        outputs=[status_box, report_box, score_box, stats_box]
    )

    # Also allow pressing Enter in the text box
    query_box.submit(
        fn=run_research,
        inputs=query_box,
        outputs=[status_box, report_box, score_box, stats_box]
    )

    # Example queries to get started quickly
    gr.Examples(
        examples=[
            ["What are the latest breakthroughs in fusion energy?"],
            ["How is AI changing drug discovery in 2024?"],
            ["What is the current state of quantum computing?"],
            ["What are the most promising cancer treatments emerging right now?"],
        ],
        inputs=query_box
    )

if __name__ == "__main__":
    demo.launch()
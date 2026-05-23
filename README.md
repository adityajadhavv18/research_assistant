# Research Assistant

An agentic AI research assistant built with LangGraph, GPT-4o-mini, Tavily, and Gradio. Ask any research question and the agent autonomously plans, searches the web, synthesises findings, and produces a structured report — all in real time.

---

## What it does

1. **Plans** — breaks your question into 3 focused sub-questions using an LLM
2. **Researches** — searches the web for each sub-question using the Tavily API
3. **Evaluates** — the LLM judges whether the results are sufficient before writing
4. **Writes** — synthesises all findings into a structured report with sources
5. **Self-grades** — scores the report quality out of 10 with one-line feedback

---

## Project structure

```
research_assistant/
├── .env                  # your API keys (never commit this)
├── requirements.txt      # all dependencies
├── phase1_basics.py      # LangGraph fundamentals — state, nodes, edges
├── phase2_agent.py       # real LLM calls + web search + research loop
├── phase3_writer.py      # structured output + LLM quality check node
└── app.py                # Gradio UI with live streaming output
```

---

## Tech stack

| Tool | Purpose |
|------|---------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent graph engine — nodes, edges, state management |
| [LangChain OpenAI](https://python.langchain.com) | GPT-4o-mini wrapper for LLM calls |
| [Tavily](https://tavily.com) | Real-time web search API built for AI agents |
| [Gradio](https://gradio.app) | Web UI with live streaming support |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | API key management |

---

## Getting started

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/research-assistant.git
cd research-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-dev-...
```

Getting your keys:
- **OpenAI** — [platform.openai.com](https://platform.openai.com) → API keys
- **Tavily** — [tavily.com](https://tavily.com) → sign up for free (1,000 credits/month)

### 5. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## Requirements

```
langgraph
langchain-openai
tavily-python
gradio
python-dotenv
```

Or install all at once:

```bash
pip install langgraph langchain-openai tavily-python gradio python-dotenv
```

---

## How the agent works

The agent is a LangGraph state graph. Every node reads from a shared `ResearchState` dictionary and writes updates back to it. Nodes never call each other directly — they communicate only through state.

```
User query
    │
    ▼
planner_node          → LLM breaks query into 3 sub-questions
    │
    ▼
researcher_node ◄─────────────────────────────────┐
    │                                               │
    ▼                                               │
should_continue?  ── more to search ───────────────┘
    │
    └── done ──►  writer_node       → LLM writes structured report
                      │
                      ▼
               quality_check_node  → LLM scores report 1–10
                      │
                      ▼
                     END
```

### State schema

```python
class ResearchState(TypedDict):
    query:            str    # original user question
    sub_questions:    list   # planner breaks query into these
    results:          list   # search results accumulate here
    report:           str    # final synthesised report
    loop_count:       int    # tracks how many searches done
    quality_score:    int    # LLM scores the report 1–10
    quality_feedback: str    # one-sentence feedback on the report
```

### LLM calls per run

| Node | LLM call | Purpose |
|------|----------|---------|
| `planner_node` | 1 | Break query into sub-questions |
| `should_continue` | 1 | Judge if results are sufficient |
| `writer_node` | 1 | Synthesise results into report |
| `quality_check_node` | 1 | Score the report |
| **Total** | **4** | |

### Tavily calls per run

3 calls by default — one per sub-question. May increase by 1 if the LLM quality check returns NO (agent adds a follow-up search automatically).

---

## Gradio UI

The UI uses `graph.stream()` instead of `graph.invoke()` so each component updates live as nodes complete — you see progress in real time rather than waiting for the full run to finish.

| Component | Updates after |
|-----------|--------------|
| Agent status | Every node |
| Report | `writer_node` completes |
| Quality score | `quality_check_node` completes |
| Run stats | `quality_check_node` completes |

---

## Learning path

This project was built in 4 phases, each introducing one new concept:

| Phase | File | What you learn |
|-------|------|---------------|
| 1 | `phase1_basics.py` | State, nodes, edges — LangGraph fundamentals |
| 2 | `phase2_agent.py` | LLM calls, tool calls, conditional edges, loops |
| 3 | `phase3_writer.py` | Structured prompts, LLM self-evaluation, quality checks |
| 4 | `app.py` | Gradio UI, `graph.stream()`, live streaming |

---

## Customisation ideas

- Swap `gpt-4o-mini` for `claude-sonnet-4-5` — just change the LangChain import and model name
- Increase `max_results` in `researcher_node` for more thorough research (uses more Tavily credits)
- Add a `fact_checker_node` after the writer that verifies claims against the source URLs
- Save reports to a local file or database by adding a `save_node` before `END`
- Change `temperature=0` to `0.3` in the writer for slightly more varied report prose

---

## Common issues

**Report not showing in UI**
Make sure `report_box` is defined as `gr.Markdown(value="*Report will appear here...*")` — the `label` parameter alone can prevent markdown from rendering.

**Results collected: 0**
Use `final_state.update(state)` instead of `final_state = state` inside the stream loop — the latter replaces the whole dict and loses accumulated results.

**Tavily returns no results**
Your query may be too vague. The planner handles this automatically, but you can also try a more specific question.

**OpenAI rate limit errors**
You are on a free-tier OpenAI key. Add `time.sleep(1)` between LLM calls or upgrade to a paid plan.

---

Built by Aditya as a hands-on introduction to agentic AI with LangGraph.

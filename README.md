# 📚 AI Book Recommender

A LangGraph-powered book recommendation agent with a Gradio frontend.

## Architecture

```
User (Gradio UI)
      │
      ▼
  Gradio Frontend  (frontend/app.py)
      │
      ▼
  LangGraph Agent  (agent/graph.py)
      │
      ├── Node: agent        → open-source local Qwen-2.5-32b reasons, decides which tools to call
      ├── Node: tools        → Executes tool calls (parallel if needed)
      │     ├── google_books_search  → Structured metadata, ratings, ISBNs
      │     └── tavily_book_search   → Live web: recent releases, lists, reviews
      └── Node: synthesize   → Extracts structured recs → displayed as cards
```

## Setup

### 1. Clone & install
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required keys:**
- `OPENAI_API_KEY` — pass "dummy" because using MN5 local model
- `TAVILY_API_KEY` — from app.tavily.com (free tier available)

**Optional:**
- `GOOGLE_BOOKS_API_KEY` — from console.cloud.google.com (works without it, but rate limited)

### 3. Run
```bash
python frontend/app.py
```

Open http://localhost:7860

## Project Structure

```
book-recommender/
├── agent/
│   ├── __init__.py
│   ├── graph.py        ← LangGraph graph definition (nodes + edges)
│   └── state.py        ← AgentState TypedDict
├── tools/
│   ├── __init__.py
│   ├── google_books.py ← Google Books API tool
│   └── tavily_search.py← Tavily web search tool
├── frontend/
│   └── app.py          ← Gradio UI
├── requirements.txt
└── .env
```

## Extending the Agent

### Add a new tool
1. Create `tools/my_tool.py` with a `@tool` decorated function
2. Import it in `tools/__init__.py` and add to `ALL_TOOLS`
3. The agent will automatically learn to use it from the docstring

### Add memory / user profiles
Replace the in-memory `agent_state` in `frontend/app.py` with a persistent store
(Redis, SQLite, LangGraph's built-in checkpointing) to remember preferences across sessions.

### Add a vector store
Embed past recommendations with OpenAI embeddings → store in Chroma/FAISS →
add a `similar_books_search` tool that does semantic similarity search.

"""
LangGraph Agent Graph
Defines the nodes and edges of the book recommendation agent.

Flow:
  START → extract_preferences → call_tools (loop) → synthesize → END
"""

import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import AgentState
from tools import ALL_TOOLS

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="/gpfs/projects/bsc02/llm_models/huggingface_models/Qwen3-32B",
    openai_api_key="dummy",  # vLLM doesn't need a real key
    openai_api_base="http://localhost:8002/v1",  # Your forwarded port
    temperature=0.7,
    max_tokens=500,
    streaming=True
)

#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = """You are an expert book recommendation assistant with deep knowledge of literature across all genres.

Your job is to recommend books based on the user's preferences, mood, reading history, and interests.

You have access to two tools:
1. **google_books_search** — Use for structured metadata: specific titles, authors, ISBNs, page counts, ratings.
2. **tavily_book_search** — Use for current trends, recent releases (2023-2026), award winners, curated lists, Reddit recommendations.

Strategy:
- Always use BOTH tools to give comprehensive, current recommendations.
- Use Tavily first for discovery, then Google Books to get metadata on promising titles.
- Ask clarifying questions if the user's request is vague (genre? mood? recently read books to avoid?)
- Provide 3-5 well-reasoned recommendations with a brief explanation for each.
- Be conversational and enthusiastic about books!
"""

# ── Nodes ──────────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> AgentState:
    """
    Core reasoning node. The LLM decides whether to call tools or respond to the user.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Router: if the last message has tool calls → run tools, else → synthesize & end.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "synthesize"


def synthesize_node(state: AgentState) -> AgentState:
    """
    Final node: extracts structured recommendations from the conversation
    and stores them in state for the Gradio UI to display.
    """
    extract_prompt = """Based on the conversation so far, extract the final book recommendations 
    as a JSON array. Each item should have:
    {
        "title": "...",
        "author": "...",
        "why": "one sentence why this book fits the user",
        "genre": "...",
        "published": "...",
        "rating": "...",
        "isbn": "..."
    }
    
    Return ONLY valid JSON array, nothing else."""

    messages = state["messages"] + [HumanMessage(content=extract_prompt)]
    response = llm.invoke(messages)

    try:
        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        recommendations = json.loads(raw.strip())
    except (json.JSONDecodeError, IndexError):
        recommendations = []

    return {"recommendations": recommendations}


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_graph():
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("synthesize", synthesize_node)

    # Set entry point
    graph.set_entry_point("agent")

    # Conditional edges from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "synthesize": "synthesize",
        }
    )

    # After tools, always go back to agent
    graph.add_edge("tools", "agent")

    # Synthesize → END
    graph.add_edge("synthesize", END)

    return graph.compile()


# Singleton graph instance
recommendation_graph = build_graph()

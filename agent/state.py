"""
LangGraph State Definition
The state object is passed between all nodes in the graph.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Core state for the book recommendation agent.
    
    - messages: Full conversation history (user + assistant + tool messages)
    - user_preferences: Extracted preferences (genre, mood, length, etc.)
    - recommendations: Final curated list of book recommendations
    """
    messages: Annotated[list[BaseMessage], add_messages]
    user_preferences: dict          # e.g. {"genre": "sci-fi", "mood": "uplifting"}
    recommendations: list[dict]     # Final structured book list for the UI

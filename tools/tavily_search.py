"""
Tavily Search Tool
Used for: recent releases, "best of" lists, niche genres, Reddit recs, blog posts.
"""

import os
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class TavilyBookSearchInput(BaseModel):
    query: str


@tool("tavily_book_search", args_schema=TavilyBookSearchInput)
def tavily_book_search(query: str) -> str:
    """
    Search the web for book recommendations, recent releases, curated lists, and reviews.
    Use this for: trending books, recent publications (2023-2026), genre-specific lists,
    award winners, 'readers also liked' style queries, or anything needing current info.
    """
    search = TavilySearch(
        max_results=5,
        topic="general",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )

    # Append "book recommendation" context if not present
    if "book" not in query.lower():
        query = f"book recommendations {query}"

    try:
        result = search.invoke({"query": query})

        # TavilySearch returns a dict with 'results' key containing list of search results
        if not result or not result.get("results"):
            return f"No web results found for: {query}"

        formatted = []
        for i, item in enumerate(result.get("results", []), 1):
            formatted.append(
                f"{i}. {item.get('title', 'No title')}\n"
                f"   Source: {item.get('url', '')}\n"
                f"   {item.get('content', '')[:300]}\n"
            )

        return "\n".join(formatted)

    except Exception as e:
        return f"Tavily search error: {str(e)}"

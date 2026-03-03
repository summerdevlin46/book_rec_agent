"""
Tavily Search Tool
Used for: recent releases, "best of" lists, niche genres, Reddit recs, blog posts.
"""

import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from pydantic import BaseModel


class TavilyBookSearchInput(BaseModel):
    query: str


@tool("tavily_book_search", args_schema=TavilyBookSearchInput)
def tavily_book_search(query: str) -> str:
    """
    Search the web for book recommendations, recent releases, curated lists, and reviews.
    Use this for: trending books, recent publications (2023-2026), genre-specific lists,
    award winners, 'readers also liked' style queries, or anything needing current info.
    """
    search = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )

    # Append "book recommendation" context if not present
    if "book" not in query.lower():
        query = f"book recommendations {query}"

    try:
        results = search.invoke({"query": query})

        if not results:
            return f"No web results found for: {query}"

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. {result.get('title', 'No title')}\n"
                f"   Source: {result.get('url', '')}\n"
                f"   {result.get('content', '')[:300]}\n"
            )

        return "\n".join(formatted)

    except Exception as e:
        return f"Tavily search error: {str(e)}"

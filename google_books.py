"""
Google Books API Tool
Fetches structured book metadata: title, author, description, cover, ISBN, etc.
"""

import httpx
import os
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Optional


class BookSearchInput(BaseModel):
    query: str
    max_results: int = 5


@tool("google_books_search", args_schema=BookSearchInput)
def google_books_search(query: str, max_results: int = 5) -> str:
    """
    Search Google Books API for book metadata.
    Use this when you need structured book info: covers, ISBNs, descriptions, authors, ratings.
    """
    api_key = os.getenv("GOOGLE_BOOKS_API_KEY", "")
    base_url = "https://www.googleapis.com/books/v1/volumes"

    params = {
        "q": query,
        "maxResults": max_results,
        "printType": "books",
        "langRestrict": "en",
    }
    if api_key:
        params["key"] = api_key

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

        books = []
        for item in data.get("items", []):
            info = item.get("volumeInfo", {})
            book = {
                "title": info.get("title", "Unknown"),
                "authors": info.get("authors", ["Unknown"]),
                "description": info.get("description", "No description available.")[:300],
                "published_date": info.get("publishedDate", "Unknown"),
                "page_count": info.get("pageCount", "Unknown"),
                "categories": info.get("categories", []),
                "average_rating": info.get("averageRating", "No rating"),
                "ratings_count": info.get("ratingsCount", 0),
                "thumbnail": info.get("imageLinks", {}).get("thumbnail", ""),
                "isbn": next(
                    (id["identifier"] for id in info.get("industryIdentifiers", [])
                     if id["type"] == "ISBN_13"), "Unknown"
                ),
                "preview_link": info.get("previewLink", ""),
            }
            books.append(book)

        if not books:
            return f"No books found for query: {query}"

        # Format for LLM consumption
        formatted = []
        for i, b in enumerate(books, 1):
            formatted.append(
                f"{i}. **{b['title']}** by {', '.join(b['authors'])}\n"
                f"   Published: {b['published_date']} | Pages: {b['page_count']} | "
                f"Rating: {b['average_rating']} ({b['ratings_count']} reviews)\n"
                f"   Categories: {', '.join(b['categories']) or 'General'}\n"
                f"   Description: {b['description']}\n"
                f"   ISBN: {b['isbn']} | Preview: {b['preview_link']}\n"
            )

        return "\n".join(formatted)

    except httpx.HTTPError as e:
        return f"Error fetching from Google Books: {str(e)}"

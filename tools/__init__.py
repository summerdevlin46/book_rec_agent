"""Utility tools used by the book recommendation agent.

This package exports all tools for easy discovery by the agent graph.
"""

# Import individual tools so they are available when the package is
# imported. The README notes that tools should be added here and
# included in ALL_TOOLS if necessary. We'll define ALL_TOOLS lazily.

"""Utility tools used by the book recommendation agent.

This package exports all tools for easy discovery by the agent graph.
"""

# Import the tools so they are available when importing the package.
from .google_books import google_books_search
from .tavily_search import tavily_book_search

# The list of all tool callables; the agent graph can import this
# variable if it needs to iterate over available tools.
ALL_TOOLS = [google_books_search, tavily_book_search]

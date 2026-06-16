from langchain_core.tools import tool
from typing import Optional
from app.utils.logger import logger

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

@tool
def web_search_tool(query: str, max_results: int = 3) -> str:
    """
    Search the web for current information using DuckDuckGo.
    Use this when the local database lacks information.
    Returns the retrieved search results as a formatted string.
    
    Args:
        query (str): The specific search query.
        max_results (int, optional): The maximum number of results to return. Defaults to 3.
    """
    logger.info(f"Web search tool invoked with query: '{query}', max_results: {max_results}")
    
    if not DDGS:
        logger.warning("duckduckgo-search is not installed.")
        return "Error: duckduckgo-search is not installed. Please pip install duckduckgo-search."
        
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"No results found on the web for query: '{query}'."
            
            formatted_results = []
            for i, r in enumerate(results):
                formatted_results.append(
                    f"--- Web Result {i+1} ---\n"
                    f"Title: {r.get('title')}\n"
                    f"URL: {r.get('href')}\n"
                    f"Snippet:\n{r.get('body')}\n"
                )
            
            return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Web search tool failed: {e}")
        return f"Error executing web search: {str(e)}"

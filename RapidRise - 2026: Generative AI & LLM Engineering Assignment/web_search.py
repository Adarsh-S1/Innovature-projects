"""
web_search.py — DuckDuckGo web search tool for out-of-scope queries.
"""

import json
from duckduckgo_search import DDGS
import llm_client


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, body, href
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        print(f"[WebSearch] Found {len(results)} results for: '{query}'")
        return results
    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        return []


def generate_web_answer(query: str, max_results: int = 5) -> dict:
    """
    Search the web and generate an LLM-summarized answer.

    Args:
        query: The user's question.
        max_results: Number of web results to fetch.

    Returns:
        dict with structured JSON response.
    """
    # Step 1: Web search
    results = search_web(query, max_results=max_results)

    if not results:
        return {
            "chain_of_thought": "Web search returned no results.",
            "summary": "I could not find relevant information from web search.",
            "key_entities": [],
            "confidence_score": 0.0,
            "sources": [],
        }

    # Step 2: Build context from web results
    context_parts = []
    sources = []
    for r in results:
        title = r.get("title", "Unknown")
        body = r.get("body", r.get("snippet", ""))
        href = r.get("href", r.get("link", ""))
        context_parts.append(f"[Source: {title}]\nURL: {href}\n{body}")
        sources.append({"title": title, "url": href})

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Generate answer with LLM
    result = llm_client.generate(query=query, context=context)
    result["sources"] = sources
    result["search_type"] = "web_search"

    return result


if __name__ == "__main__":
    print("Testing web search...")
    result = generate_web_answer("What are the latest developments in quantum computing in 2025?")
    print(json.dumps(result, indent=2, default=str))

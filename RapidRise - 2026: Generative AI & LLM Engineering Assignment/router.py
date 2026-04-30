"""
router.py — Agentic intent router using LLM classification.

Routes queries to either the RAG pipeline (document-related)
or web search (general knowledge / current events).
"""

import json
import llm_client
import rag_pipeline
import web_search
import config


def route_query(query: str) -> dict:
    """
    Route a query to the appropriate tool based on LLM intent classification.

    Args:
        query: The user's question.

    Returns:
        dict with the generated response and routing metadata.
    """
    print(f"\n{'='*70}")
    print(f"[Router] Processing query: \"{query}\"")
    print(f"{'='*70}")

    # Step 1: Classify intent
    classification = llm_client.classify_intent(query, config.DOCUMENT_TOPICS)
    route = classification.get("route", "rag")
    print(f"[Router] Classification result: {json.dumps(classification)}")

    # Step 2: Route to appropriate tool
    if route == "web_search":
        search_query = classification.get("search_query", query)
        print(f"[Router] Routing to WEB SEARCH with query: '{search_query}'")
        result = web_search.generate_web_answer(search_query)
        result["route_used"] = "web_search"
    else:
        print(f"[Router] Routing to RAG PIPELINE")
        result = rag_pipeline.generate_rag_answer(query)
        result["route_used"] = "rag"

    result["original_query"] = query
    result["classification"] = classification

    return result


def demo_routing():
    """Demonstrate routing with two distinct query types."""
    print("\n" + "="*70)
    print("  AGENTIC ROUTING DEMONSTRATION")
    print("="*70)

    # Query 1: Document-related (should route to RAG)
    q1 = "What is the key innovation in the Transformer architecture described in 'Attention Is All You Need'?"
    result1 = route_query(q1)
    print(f"\n--- RESULT (RAG) ---")
    print(f"Route: {result1['route_used']}")
    print(f"Summary: {result1.get('summary', 'N/A')}")
    print(f"Confidence: {result1.get('confidence_score', 'N/A')}")
    print(f"Sources: {result1.get('sources', [])}")

    print("\n" + "-"*70)

    # Query 2: General knowledge (should route to Web Search)
    q2 = "What was the latest breakthrough in quantum computing in 2025?"
    result2 = route_query(q2)
    print(f"\n--- RESULT (Web Search) ---")
    print(f"Route: {result2['route_used']}")
    print(f"Summary: {result2.get('summary', 'N/A')}")
    print(f"Confidence: {result2.get('confidence_score', 'N/A')}")

    return result1, result2


if __name__ == "__main__":
    demo_routing()

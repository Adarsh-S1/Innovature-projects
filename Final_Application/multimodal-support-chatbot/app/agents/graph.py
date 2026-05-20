"""
LangGraph graph definition — wires up the 5-agent pipeline.
Stub implementation; agent node functions will be implemented in Part 4.
"""

from langgraph.graph import END, StateGraph

from app.agents.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Stub Agent Nodes ─────────────────────────────────────────────────────
# These will be replaced with real implementations in Part 4.


def context_router_agent(state: AgentState) -> dict:
    """Agent 1: Parse query intent and route."""
    logger.info("agent_invoked", agent="context_router", query=state.get("user_query"))
    return {
        "query_intent": "multimodal",
        "needs_image": True,
        "query_keywords": [],
        "query_concepts": [],
        "query_type": "general",
        "reformulated_query": state.get("user_query", ""),
        "bm25_weight": 0.4,
        "vector_weight": 0.6,
    }


def hybrid_search_agent(state: AgentState) -> dict:
    """Agent 2: Execute hybrid BM25 + vector search."""
    logger.info("agent_invoked", agent="hybrid_search")
    return {
        "retrieved_chunks": [],
        "rerank_scores": [],
    }


def visual_specialist_agent(state: AgentState) -> dict:
    """Agent 3: Cross-modal image retrieval via CLIP."""
    logger.info("agent_invoked", agent="visual_specialist")
    return {
        "retrieved_images": [],
    }


def answer_synthesizer_agent(state: AgentState) -> dict:
    """Agent 4: Synthesize final answer from chunks + images."""
    logger.info("agent_invoked", agent="answer_synthesizer")
    return {
        "draft_answer": "This is a stub answer.",
        "cited_sources": [],
    }


def quality_guard_agent(state: AgentState) -> dict:
    """Agent 5: Score answer quality and decide approve/loopback."""
    logger.info("agent_invoked", agent="quality_guard")
    retry_count = state.get("retry_count", 0)
    return {
        "quality_score": 0.85,
        "quality_issues": [],
        "final_answer": state.get("draft_answer", ""),
        "retry_count": retry_count,
        "response_images": state.get("retrieved_images", []),
        "response_metadata": {
            "query_intent": state.get("query_intent"),
            "retry_count": retry_count,
        },
    }


# ── Routing Functions ────────────────────────────────────────────────────


def route_after_search(state: AgentState) -> str:
    """After hybrid search, run visual specialist only if images are needed."""
    if state.get("needs_image", False):
        return "visual_specialist"
    return "answer_synthesizer"


def route_after_quality(state: AgentState) -> str:
    """After quality guard, loopback or finish."""
    score = state.get("quality_score", 1.0)
    retries = state.get("retry_count", 0)
    if score < 0.75 and retries < 2:
        return "hybrid_search"
    return END


# ── Graph Construction ───────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("context_router", context_router_agent)
    workflow.add_node("hybrid_search", hybrid_search_agent)
    workflow.add_node("visual_specialist", visual_specialist_agent)
    workflow.add_node("answer_synthesizer", answer_synthesizer_agent)
    workflow.add_node("quality_guard", quality_guard_agent)

    # Define edges
    workflow.set_entry_point("context_router")
    workflow.add_edge("context_router", "hybrid_search")

    # Conditional: run visual_specialist only if image needed
    workflow.add_conditional_edges(
        "hybrid_search",
        route_after_search,
        {
            "visual_specialist": "visual_specialist",
            "answer_synthesizer": "answer_synthesizer",
        },
    )

    workflow.add_edge("visual_specialist", "answer_synthesizer")
    workflow.add_edge("answer_synthesizer", "quality_guard")

    # Quality gate: loopback or end
    workflow.add_conditional_edges(
        "quality_guard",
        route_after_quality,
        {
            "hybrid_search": "hybrid_search",
            END: END,
        },
    )

    return workflow


# Pre-build the graph (compiled at import time for re-use)
graph = build_graph()

logger.info("langgraph_initialized", nodes=list(graph.nodes.keys()) if hasattr(graph, 'nodes') else [])

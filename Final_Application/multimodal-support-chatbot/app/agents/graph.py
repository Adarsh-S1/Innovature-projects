"""
LangGraph graph definition — wires up the 5-agent pipeline.

Graph topology:
  context_router → hybrid_search → [visual_specialist?] → answer_synthesizer
  → quality_guard → [loopback to hybrid_search | END]
"""

from langgraph.graph import END, StateGraph

from app.agents.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Import real agent implementations ────────────────────────────────────

from app.agents.context_router import context_router
from app.agents.hybrid_search import hybrid_search
from app.agents.visual_specialist import visual_specialist
from app.agents.answer_synthesizer import answer_synthesizer
from app.agents.quality_guard import quality_guard


# ── Sync wrappers for LangGraph nodes ────────────────────────────────────
# LangGraph expects sync or async; we wrap async agents uniformly.

import asyncio


def _run_async(coro):
    """Run an async coroutine from a sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — use a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def context_router_agent(state: AgentState) -> dict:
    """Agent 1: Parse query intent and route."""
    logger.info("agent_invoked", agent="context_router", query=state.get("user_query"))
    return _run_async(context_router(state))


def hybrid_search_agent(state: AgentState) -> dict:
    """Agent 2: Execute hybrid BM25 + vector search."""
    logger.info("agent_invoked", agent="hybrid_search")
    return _run_async(hybrid_search(state))


def visual_specialist_agent(state: AgentState) -> dict:
    """Agent 3: Cross-modal image retrieval via CLIP."""
    logger.info("agent_invoked", agent="visual_specialist")
    return _run_async(visual_specialist(state))


def answer_synthesizer_agent(state: AgentState) -> dict:
    """Agent 4: Synthesize final answer from chunks + images."""
    logger.info("agent_invoked", agent="answer_synthesizer")
    return _run_async(answer_synthesizer(state))


def quality_guard_agent(state: AgentState) -> dict:
    """Agent 5: Score answer quality and decide approve/loopback."""
    logger.info("agent_invoked", agent="quality_guard")
    return _run_async(quality_guard(state))


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
        logger.info(
            "quality_loopback_triggered",
            score=score,
            retry_count=retries,
        )
        return "hybrid_search"
    return END


# ── Graph Construction ───────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct and return the LangGraph workflow (not yet compiled)."""
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

logger.info(
    "langgraph_initialized",
    nodes=list(graph.nodes.keys()) if hasattr(graph, 'nodes') else [],
)

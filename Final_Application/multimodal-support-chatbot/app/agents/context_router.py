"""
Agent 1: Context Router — Parses query intent and routes to appropriate pipeline path.
Stub implementation; full logic will be added in Part 4.
"""

from app.agents.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)


async def context_router(state: AgentState) -> dict:
    """
    Parse user query for intent classification.

    Responsibilities:
    - Detect if query requires text, visual, or both
    - Extract structured query parameters (keywords, concepts, product IDs)
    - Determine if query is a follow-up (use conversation history)
    - Set dynamic BM25/vector weights based on query type

    Will use GPT-4o with structured output in Part 4.
    """
    logger.info(
        "context_router_invoked",
        query=state.get("user_query"),
        session_id=state.get("session_id"),
    )
    # Stub — returns default multimodal routing
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

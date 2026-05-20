"""
Agent 2: Hybrid Search — Executes BM25 + vector search with RRF fusion.
Stub implementation; full logic will be added in Part 3.
"""

from app.agents.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)


async def hybrid_search(state: AgentState) -> dict:
    """
    Execute hybrid search combining BM25 keyword and vector semantic search.

    Responsibilities:
    - BM25 keyword search on Milvus text collection
    - Dense vector search using query embedding
    - Reciprocal Rank Fusion (RRF) to merge results
    - Cross-encoder re-ranking for final top-K

    Will be fully implemented in Part 3.
    """
    logger.info(
        "hybrid_search_invoked",
        keywords=state.get("query_keywords"),
        intent=state.get("query_intent"),
    )
    return {
        "retrieved_chunks": [],
        "rerank_scores": [],
    }

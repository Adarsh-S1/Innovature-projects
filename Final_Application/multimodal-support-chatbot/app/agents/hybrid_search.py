"""
Agent 2: Hybrid Search — Executes BM25 + dense vector search with RRF fusion,
then re-ranks results for the top-K context passages.

Connects to the HybridSearcher and Reranker modules from Part 3.
"""

from typing import Any, Dict, List

from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.domain import QueryType

logger = get_logger(__name__)


async def hybrid_search(state: AgentState) -> dict:
    """
    Agent 2: Execute hybrid search combining BM25 keyword and vector semantic search.

    Pipeline:
    1. Embed the reformulated query using text-embedding-3-large
    2. Run BM25 + vector search via HybridSearcher (RRF fusion)
    3. Re-rank results with the Reranker (heuristic or LLM)
    4. Return top-K chunks with scores
    """
    query = state.get("reformulated_query") or state.get("user_query", "")
    query_type_str = state.get("query_type", "general")
    bm25_weight = state.get("bm25_weight", 0.4)
    vector_weight = state.get("vector_weight", 0.6)

    logger.info(
        "hybrid_search_invoked",
        query_len=len(query),
        query_type=query_type_str,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )

    settings = get_settings()

    # Map string to QueryType enum
    try:
        query_type = QueryType(query_type_str)
    except ValueError:
        query_type = QueryType.GENERAL

    # ── Step 1: Generate query embedding ─────────────────────────────
    query_vector = await _embed_query(query)

    if not query_vector:
        logger.warning("query_embedding_failed_returning_empty")
        return {
            "retrieved_chunks": [],
            "rerank_scores": [],
        }

    # ── Step 2: Hybrid search (BM25 + Vector + RRF) ──────────────────
    from app.retrieval.hybrid_searcher import HybridSearcher

    searcher = HybridSearcher()

    try:
        search_results = searcher.search(
            query=query,
            query_vector=query_vector,
            query_type=query_type,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            top_k=settings.TEXT_SEARCH_TOP_K,
        )
    except Exception as e:
        logger.error("hybrid_search_failed", error=str(e))
        search_results = []

    # ── Step 3: Rerank ───────────────────────────────────────────────
    from app.retrieval.reranker import Reranker

    reranker = Reranker()

    try:
        reranked = reranker.rerank(
            query=query,
            results=search_results,
            top_k=settings.TEXT_RERANK_TOP_K,
            use_llm=False,  # Use heuristic by default for speed
        )
    except Exception as e:
        logger.warning("reranking_failed_using_raw_results", error=str(e))
        reranked = []

    # ── Step 4: Convert to state dict format ─────────────────────────
    retrieved_chunks: List[Dict[str, Any]] = []
    rerank_scores: List[float] = []

    for r in reranked:
        retrieved_chunks.append({
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "text": r.text,
            "page_start": r.page_start,
            "page_end": r.page_end,
            "section_path": r.section_path,
            "source_file": r.source_file,
            "product_id": r.product_id,
            "linked_images": r.linked_images,
            "final_score": r.final_score,
        })
        rerank_scores.append(r.final_score)

    logger.info(
        "hybrid_search_completed",
        total_results=len(retrieved_chunks),
        top_score=rerank_scores[0] if rerank_scores else 0,
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "rerank_scores": rerank_scores,
    }


async def _embed_query(query: str) -> List[float] | None:
    """Generate embedding for the query text using shared embedder singleton."""
    try:
        from app.core.shared import get_text_embedder
        embedder = get_text_embedder()
        return embedder.embed_text(query)
    except Exception as e:
        logger.error("query_embedding_failed", error=str(e))
        return None

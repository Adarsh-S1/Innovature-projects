"""
Reranker — re-scores hybrid search results for final quality filtering.

Provides two reranking strategies:
1. LLM-based reranking via GPT-4o (highest quality, higher latency)
2. Lightweight heuristic reranking (no API call, low latency)

Also enforces the minimum similarity threshold from the spec — results
below 0.35 are dropped to prevent hallucinated answers from irrelevant context.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.exceptions import RerankingError
from app.core.logging import get_logger
from app.retrieval.hybrid_searcher import SearchResult

logger = get_logger(__name__)

# Minimum relevance threshold — below this, results are dropped
_MIN_RELEVANCE_THRESHOLD = 0.35


@dataclass
class RerankResult:
    """A reranked search result with original and reranked scores."""

    chunk_id: str
    text: str
    original_rrf_score: float
    rerank_score: float
    final_score: float
    doc_id: str = ""
    page_start: int = 0
    page_end: int = 0
    section_path: List[str] = None
    source_file: str = ""
    product_id: str = ""
    linked_images: List[str] = None

    def __post_init__(self):
        if self.section_path is None:
            self.section_path = []
        if self.linked_images is None:
            self.linked_images = []


class Reranker:
    """
    Re-scores and filters search results to improve final answer quality.
    """

    def __init__(self):
        self._settings = get_settings()
        self._openai_client = None

    def _ensure_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self._settings.OPENAI_API_KEY,
                max_retries=self._settings.OPENAI_MAX_RETRIES,
                timeout=self._settings.OPENAI_TIMEOUT,
            )

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        use_llm: bool = False,
    ) -> List[RerankResult]:
        """
        Rerank search results for final quality.

        Args:
            query: Original user query.
            results: Search results from HybridSearcher.
            top_k: Max results to return after reranking.
            use_llm: If True, use LLM-based reranking. Otherwise, use
                     heuristic scoring.

        Returns:
            Filtered and reranked list of RerankResult objects.
        """
        top_k = top_k or self._settings.TEXT_RERANK_TOP_K

        if not results:
            return []

        logger.info(
            "reranking_started",
            input_count=len(results),
            method="llm" if use_llm else "heuristic",
            top_k=top_k,
        )

        if use_llm:
            reranked = self._llm_rerank(query, results)
        else:
            reranked = self._heuristic_rerank(query, results)

        # Filter by minimum relevance threshold
        filtered = [
            r for r in reranked
            if r.final_score >= _MIN_RELEVANCE_THRESHOLD
        ]

        # Sort by final_score descending and take top_k
        filtered.sort(key=lambda r: r.final_score, reverse=True)
        final = filtered[:top_k]

        logger.info(
            "reranking_completed",
            input_count=len(results),
            above_threshold=len(filtered),
            returned=len(final),
        )

        return final

    def _heuristic_rerank(
        self, query: str, results: List[SearchResult]
    ) -> List[RerankResult]:
        """
        Lightweight heuristic reranking based on:
        - RRF score (already fused BM25 + vector)
        - Query term overlap ratio
        - Section depth bonus (deeper = more specific = often better)
        """
        import re

        query_terms = set(re.findall(r'\b\w+\b', query.lower()))

        reranked: List[RerankResult] = []

        for result in results:
            # Term overlap score
            doc_terms = set(re.findall(r'\b\w+\b', result.text.lower()))
            if doc_terms:
                overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
            else:
                overlap = 0.0

            # Section depth bonus: deeper section = more specific
            section_depth = len(result.section_path)
            depth_bonus = min(section_depth * 0.05, 0.15)

            # Normalize RRF score to 0–1 range (approximate)
            # RRF scores are typically small; scale up
            normalized_rrf = min(result.rrf_score * 100, 1.0)

            # Composite rerank score
            rerank_score = (
                0.60 * normalized_rrf
                + 0.30 * overlap
                + 0.10 * depth_bonus
            )

            # Final score blends original and rerank
            final_score = 0.5 * normalized_rrf + 0.5 * rerank_score

            reranked.append(RerankResult(
                chunk_id=result.chunk_id,
                text=result.text,
                original_rrf_score=result.rrf_score,
                rerank_score=rerank_score,
                final_score=final_score,
                doc_id=result.doc_id,
                page_start=result.page_start,
                page_end=result.page_end,
                section_path=result.section_path,
                source_file=result.source_file,
                product_id=result.product_id,
                linked_images=result.linked_images,
            ))

        return reranked

    def _llm_rerank(
        self, query: str, results: List[SearchResult]
    ) -> List[RerankResult]:
        """
        LLM-based reranking: ask GPT-4o to score relevance of each
        candidate passage against the query.

        Falls back to heuristic reranking on API failure.
        """
        self._ensure_openai_client()

        try:
            # Build passages for LLM scoring
            passages = []
            for i, r in enumerate(results[:20]):  # Limit to 20 for cost
                passages.append(
                    f"[{i}] (Source: {r.source_file}, Pages {r.page_start}-{r.page_end})\n"
                    f"{r.text[:500]}"
                )

            passages_text = "\n\n".join(passages)

            response = self._openai_client.chat.completions.create(
                model=self._settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a search relevance judge. Given a query and "
                            "a list of passages, score each passage's relevance "
                            "to the query on a scale of 0.0 to 1.0.\n\n"
                            "Respond ONLY with a JSON array of objects: "
                            '[{"index": 0, "score": 0.95}, ...]'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Query: {query}\n\n"
                            f"Passages:\n{passages_text}"
                        ),
                    },
                ],
                max_tokens=500,
                temperature=0.0,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            import json
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0]

            scores = json.loads(content)
            score_map = {s["index"]: s["score"] for s in scores}

            reranked: List[RerankResult] = []
            for i, result in enumerate(results[:20]):
                llm_score = score_map.get(i, 0.0)
                normalized_rrf = min(result.rrf_score * 100, 1.0)

                # Blend LLM score with original RRF
                final_score = 0.6 * llm_score + 0.4 * normalized_rrf

                reranked.append(RerankResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    original_rrf_score=result.rrf_score,
                    rerank_score=llm_score,
                    final_score=final_score,
                    doc_id=result.doc_id,
                    page_start=result.page_start,
                    page_end=result.page_end,
                    section_path=result.section_path,
                    source_file=result.source_file,
                    product_id=result.product_id,
                    linked_images=result.linked_images,
                ))

            return reranked

        except Exception as e:
            logger.warning(
                "llm_rerank_failed_falling_back",
                error=str(e),
            )
            return self._heuristic_rerank(query, results)

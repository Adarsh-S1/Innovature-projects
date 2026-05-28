"""
Hybrid Searcher — BM25 + dense vector search with Reciprocal Rank Fusion (RRF).

Combines sparse keyword search (BM25 via in-memory index) with dense
semantic search (Milvus HNSW) using RRF, which normalises rank differences
across scoring scales without manual score calibration.

Dynamic weight adjustment is driven by the Context Router's QueryType:
- Error codes / Part numbers  → BM25-heavy  (0.65 / 0.35)
- How-to / Procedures         → Vector-heavy (0.25 / 0.75)
- Component location          → Vector-heavy (0.30 / 0.70)
- General concept             → Vector-heavy (0.20 / 0.80)
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.exceptions import VectorSearchError
from app.core.logging import get_logger
from app.models.domain import QueryType

logger = get_logger(__name__)


# ── Dynamic weight table (from Part 3 spec) ─────────────────────────────

_WEIGHT_TABLE: Dict[QueryType, Tuple[float, float]] = {
    QueryType.TROUBLESHOOT: (0.65, 0.35),    # error codes, part numbers
    QueryType.HOW_TO: (0.25, 0.75),           # procedural queries
    QueryType.LOCATE_COMPONENT: (0.30, 0.70), # component location
    QueryType.SPECIFICATION: (0.50, 0.50),    # balanced for specs
    QueryType.GENERAL: (0.20, 0.80),          # conceptual queries
}


@dataclass
class SearchResult:
    """A single search result with fused scoring."""

    chunk_id: str
    doc_id: str = ""
    text: str = ""
    page_start: int = 0
    page_end: int = 0
    section_path: List[str] = field(default_factory=list)
    chunk_type: str = ""
    source_file: str = ""
    product_id: str = ""
    language: str = "en"
    linked_images: List[str] = field(default_factory=list)
    rrf_score: float = 0.0
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    vector_distance: Optional[float] = None


class BM25Index:
    """
    Lightweight in-memory BM25 index for keyword search.

    Built from Milvus text collection data. Recomputed per-query
    to avoid maintaining a persistent index (viable for collections
    up to ~100k chunks; for larger scale, swap for Milvus sparse vectors).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._docs: Dict[str, List[str]] = {}       # chunk_id → tokens
        self._doc_lengths: Dict[str, int] = {}
        self._avg_dl: float = 0.0
        self._idf: Dict[str, float] = {}
        self._df: Dict[str, int] = defaultdict(int)  # document frequency
        self._n: int = 0

    def build(self, documents: Dict[str, str]) -> None:
        """
        Build BM25 index from {chunk_id: text} mapping.

        Args:
            documents: dict mapping chunk_id to raw text.
        """
        self._n = len(documents)
        if self._n == 0:
            return

        total_length = 0
        self._df = defaultdict(int)

        for chunk_id, text in documents.items():
            tokens = self._tokenize(text)
            self._docs[chunk_id] = tokens
            self._doc_lengths[chunk_id] = len(tokens)
            total_length += len(tokens)

            # Count unique terms per document
            for term in set(tokens):
                self._df[term] += 1

        self._avg_dl = total_length / self._n if self._n > 0 else 0

        # Pre-compute IDF for all terms
        import math
        for term, df in self._df.items():
            self._idf[term] = math.log(
                (self._n - df + 0.5) / (df + 0.5) + 1.0
            )

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Score all documents against the query.

        Returns:
            Sorted list of (chunk_id, bm25_score), descending.
        """
        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = {}

        for chunk_id, doc_tokens in self._docs.items():
            score = 0.0
            dl = self._doc_lengths[chunk_id]

            # Count term frequencies in this document
            tf_map: Dict[str, int] = defaultdict(int)
            for t in doc_tokens:
                tf_map[t] += 1

            for qt in query_tokens:
                if qt not in self._idf:
                    continue

                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue

                idf = self._idf[qt]
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * (dl / max(self._avg_dl, 1))
                )
                score += idf * (numerator / denominator)

            if score > 0:
                scores[chunk_id] = score

        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer, lowercased."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens


class HybridSearcher:
    """
    Hybrid search combining BM25 keyword search with Milvus dense vector
    search, fused using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        self._settings = get_settings()
        self._bm25_index = BM25Index()
        self._text_corpus_loaded = False

    def search(
        self,
        query: str,
        query_vector: List[float],
        query_type: QueryType = QueryType.GENERAL,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
        top_k: Optional[int] = None,
        doc_id_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Execute hybrid search: BM25 + vector search → RRF fusion.

        Args:
            query: Raw user query string.
            query_vector: 3072-dim embedding of the query.
            query_type: Type of query (determines dynamic weights).
            bm25_weight: Override BM25 weight (0–1). None = use dynamic table.
            vector_weight: Override vector weight. None = use dynamic table.
            top_k: Number of final results to return.
            doc_id_filter: Restrict to a specific document.
            language_filter: Restrict to a specific language.

        Returns:
            List of SearchResult objects, ranked by RRF score.
        """
        top_k = top_k or self._settings.TEXT_SEARCH_TOP_K

        # Resolve weights
        if bm25_weight is not None and vector_weight is not None:
            w_bm25, w_vec = bm25_weight, vector_weight
        else:
            w_bm25, w_vec = _WEIGHT_TABLE.get(
                query_type, (0.40, 0.60)
            )

        logger.info(
            "hybrid_search_started",
            query_len=len(query),
            query_type=query_type.value,
            bm25_weight=w_bm25,
            vector_weight=w_vec,
            top_k=top_k,
        )

        # ── Step 1: Load text corpus for BM25 if not cached ──────────────
        text_corpus = self._load_text_corpus(doc_id_filter, language_filter)

        # ── Step 2: BM25 search ──────────────────────────────────────────
        bm25_results = self._bm25_search(query, text_corpus, top_k=top_k * 2)

        # ── Step 3: Vector search via Milvus ─────────────────────────────
        vector_results = self._vector_search(
            query_vector, top_k=top_k * 2,
            doc_id_filter=doc_id_filter, language_filter=language_filter,
        )

        # ── Step 4: RRF fusion ───────────────────────────────────────────
        fused = self._reciprocal_rank_fusion(
            bm25_results, vector_results,
            bm25_weight=w_bm25, vector_weight=w_vec,
        )

        # ── Step 5: Assemble final results ───────────────────────────────
        results = self._assemble_results(fused, text_corpus, top_k)

        logger.info(
            "hybrid_search_completed",
            bm25_hits=len(bm25_results),
            vector_hits=len(vector_results),
            fused_results=len(results),
        )

        return results

    def _load_text_corpus(
        self, doc_id_filter: Optional[str], language_filter: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load text chunks from Milvus for BM25 indexing.
        Returns dict: chunk_id → {text, doc_id, page_start, ...}
        """
        from app.db.milvus_client import milvus_manager

        collection = milvus_manager.text_collection
        if collection is None:
            logger.warning("milvus_text_collection_unavailable")
            return {}

        try:
            collection.load()

            # Build filter expression (sanitized to prevent injection)
            from app.core.sanitize import build_milvus_filter
            expr = build_milvus_filter(
                doc_id=doc_id_filter, language=language_filter
            )

            results = collection.query(
                expr=expr if expr else "chunk_id != ''",
                output_fields=[
                    "chunk_id", "doc_id", "text", "page_start", "page_end",
                    "section_path", "chunk_type", "source_file", "product_id",
                    "language", "linked_images",
                ],
                limit=10000,  # Practical upper bound
            )

            corpus: Dict[str, Dict[str, Any]] = {}
            texts_for_bm25: Dict[str, str] = {}
            for r in results:
                cid = r["chunk_id"]
                corpus[cid] = r
                texts_for_bm25[cid] = r.get("text", "")

            # Rebuild BM25 index
            self._bm25_index = BM25Index()
            self._bm25_index.build(texts_for_bm25)

            logger.debug("text_corpus_loaded", count=len(corpus))
            return corpus

        except Exception as e:
            logger.error("text_corpus_load_failed", error=str(e))
            return {}

    def _bm25_search(
        self, query: str, corpus: Dict[str, Dict[str, Any]], top_k: int = 40
    ) -> List[Tuple[str, float]]:
        """Run BM25 search on the loaded corpus."""
        if not corpus:
            return []

        return self._bm25_index.search(query, top_k=top_k)

    def _vector_search(
        self,
        query_vector: List[float],
        top_k: int = 40,
        doc_id_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Run dense vector search on Milvus text_collection.

        Returns:
            List of (chunk_id, distance) tuples.
        """
        from app.db.milvus_client import milvus_manager

        collection = milvus_manager.text_collection
        if collection is None:
            return []

        try:
            collection.load()

            # Build filter expression (sanitized to prevent injection)
            from app.core.sanitize import build_milvus_filter
            expr = build_milvus_filter(
                doc_id=doc_id_filter, language=language_filter
            )

            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 128},
            }

            results = collection.search(
                data=[query_vector],
                anns_field="text_vector",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["chunk_id"],
            )

            hits = []
            if results and len(results) > 0:
                for hit in results[0]:
                    hits.append((hit.entity.get("chunk_id"), hit.distance))

            return hits

        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            return []

    @staticmethod
    def _reciprocal_rank_fusion(
        bm25_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
        k: int = 60,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> List[Tuple[str, float, Optional[int], Optional[int], Optional[float]]]:
        """
        Reciprocal Rank Fusion (RRF) to merge two ranked lists.

        RRF normalises across scoring scales:
            score(d) = Σ weight_i / (k + rank_i)

        Args:
            bm25_results: (chunk_id, bm25_score) sorted descending.
            vector_results: (chunk_id, distance) sorted by relevance.
            k: RRF constant (60 is standard).
            bm25_weight: Weight for BM25 rankings.
            vector_weight: Weight for vector rankings.

        Returns:
            List of (chunk_id, rrf_score, bm25_rank, vector_rank, distance),
            sorted by rrf_score descending.
        """
        scores: Dict[str, float] = {}
        bm25_ranks: Dict[str, int] = {}
        vector_ranks: Dict[str, int] = {}
        vector_distances: Dict[str, float] = {}

        for rank, (doc_id, _score) in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (k + rank + 1)
            bm25_ranks[doc_id] = rank + 1

        for rank, (doc_id, distance) in enumerate(vector_results):
            scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank + 1)
            vector_ranks[doc_id] = rank + 1
            vector_distances[doc_id] = distance

        # Sort by RRF score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            (
                doc_id,
                rrf_score,
                bm25_ranks.get(doc_id),
                vector_ranks.get(doc_id),
                vector_distances.get(doc_id),
            )
            for doc_id, rrf_score in sorted_results
        ]

    def _assemble_results(
        self,
        fused: List[Tuple[str, float, Optional[int], Optional[int], Optional[float]]],
        corpus: Dict[str, Dict[str, Any]],
        top_k: int,
    ) -> List[SearchResult]:
        """Convert fused RRF results into SearchResult objects."""
        results = []

        for chunk_id, rrf_score, bm25_rank, vector_rank, v_dist in fused[:top_k]:
            meta = corpus.get(chunk_id, {})

            # Parse JSON-serialized fields
            from app.core.shared import parse_json_field
            section_path = parse_json_field(meta.get("section_path", "[]"))
            linked_images = parse_json_field(meta.get("linked_images", "[]"))

            results.append(SearchResult(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                text=meta.get("text", ""),
                page_start=meta.get("page_start", 0),
                page_end=meta.get("page_end", 0),
                section_path=section_path,
                chunk_type=meta.get("chunk_type", ""),
                source_file=meta.get("source_file", ""),
                product_id=meta.get("product_id", ""),
                language=meta.get("language", "en"),
                linked_images=linked_images,
                rrf_score=rrf_score,
                bm25_rank=bm25_rank,
                vector_rank=vector_rank,
                vector_distance=v_dist,
            ))

        return results

    def search_offline(
        self,
        query: str,
        query_vector: List[float],
        chunks: List[Dict[str, Any]],
        query_type: QueryType = QueryType.GENERAL,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Offline hybrid search — operates entirely in-memory without Milvus.

        Useful for testing and for environments where Milvus is unavailable.

        Args:
            query: Raw query text.
            query_vector: Query embedding vector.
            chunks: List of dicts with keys: chunk_id, text, text_vector, ...
            query_type: Determines dynamic weights.
            top_k: Number of results to return.
        """
        if bm25_weight is not None and vector_weight is not None:
            w_bm25, w_vec = bm25_weight, vector_weight
        else:
            w_bm25, w_vec = _WEIGHT_TABLE.get(query_type, (0.40, 0.60))

        # Build BM25 index from provided chunks
        texts = {c["chunk_id"]: c.get("text", "") for c in chunks}
        bm25 = BM25Index()
        bm25.build(texts)
        bm25_results = bm25.search(query, top_k=top_k * 2)

        # Dense vector scoring
        vector_results: List[Tuple[str, float]] = []
        if query_vector:
            qv = np.array(query_vector, dtype=np.float32)
            qv_norm = np.linalg.norm(qv)
            if qv_norm > 0:
                qv = qv / qv_norm

            for c in chunks:
                cv = c.get("text_vector")
                if cv is None:
                    continue
                cv = np.array(cv, dtype=np.float32)
                cv_norm = np.linalg.norm(cv)
                if cv_norm > 0:
                    cv = cv / cv_norm
                sim = float(np.dot(qv, cv))
                vector_results.append((c["chunk_id"], sim))

            vector_results.sort(key=lambda x: x[1], reverse=True)
            vector_results = vector_results[: top_k * 2]

        # RRF fusion
        fused = self._reciprocal_rank_fusion(
            bm25_results, vector_results,
            bm25_weight=w_bm25, vector_weight=w_vec,
        )

        # Assemble
        corpus = {c["chunk_id"]: c for c in chunks}
        return self._assemble_results(fused, corpus, top_k)

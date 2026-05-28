"""
Unit tests for the HybridSearcher (offline mode) and BM25Index.

Tests cover:
- BM25 index building and search
- RRF fusion logic
- Offline hybrid search (no Milvus required)
- Weight table lookups
- Edge cases (empty results, single-term queries)
"""

import pytest

from app.retrieval.hybrid_searcher import BM25Index, HybridSearcher, _WEIGHT_TABLE
from app.models.domain import QueryType


@pytest.fixture
def bm25_index():
    """Create a BM25 index with sample documents."""
    index = BM25Index()
    documents = {
        "chunk-1": "The motherboard has four RAM slots near the CPU socket.",
        "chunk-2": "Error code E100 indicates a power supply failure.",
        "chunk-3": "To install RAM, open the retention clips and press firmly.",
        "chunk-4": "The CPU socket is located at the center of the motherboard.",
        "chunk-5": "Wiring diagram shows the connection between power supply and motherboard.",
    }
    index.build(documents)
    return index


@pytest.fixture
def sample_chunks():
    """Sample chunks with text vectors for offline search testing."""
    import numpy as np
    np.random.seed(42)
    return [
        {
            "chunk_id": "c1",
            "doc_id": "doc-1",
            "text": "The motherboard has four RAM slots near the CPU socket.",
            "page_start": 1,
            "page_end": 1,
            "section_path": '["Hardware", "Motherboard"]',
            "chunk_type": "paragraph",
            "source_file": "manual.pdf",
            "product_id": "PROD-1",
            "language": "en",
            "linked_images": "[]",
            "text_vector": np.random.randn(384).tolist(),
        },
        {
            "chunk_id": "c2",
            "doc_id": "doc-1",
            "text": "Error code E100 indicates a power supply failure.",
            "page_start": 5,
            "page_end": 5,
            "section_path": '["Troubleshooting"]',
            "chunk_type": "paragraph",
            "source_file": "manual.pdf",
            "product_id": "PROD-1",
            "language": "en",
            "linked_images": "[]",
            "text_vector": np.random.randn(384).tolist(),
        },
        {
            "chunk_id": "c3",
            "doc_id": "doc-1",
            "text": "To install RAM, open the retention clips and press firmly.",
            "page_start": 3,
            "page_end": 3,
            "section_path": '["Hardware", "Installation"]',
            "chunk_type": "paragraph",
            "source_file": "manual.pdf",
            "product_id": "PROD-1",
            "language": "en",
            "linked_images": '["img-1"]',
            "text_vector": np.random.randn(384).tolist(),
        },
    ]


class TestBM25Index:
    """Tests for the BM25Index implementation."""

    def test_build_index_succeeds(self, bm25_index):
        """BM25 index should build without errors."""
        assert bm25_index._n == 5

    def test_search_returns_results(self, bm25_index):
        """BM25 search should return results for relevant queries."""
        results = bm25_index.search("RAM slots motherboard", top_k=3)
        assert len(results) > 0
        # First result should be the chunk about RAM slots
        assert results[0][0] in ("chunk-1", "chunk-3")

    def test_search_scores_descending(self, bm25_index):
        """Results should be sorted by score in descending order."""
        results = bm25_index.search("motherboard", top_k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self, bm25_index):
        """Should return at most top_k results."""
        results = bm25_index.search("motherboard", top_k=2)
        assert len(results) <= 2

    def test_search_irrelevant_query(self, bm25_index):
        """Irrelevant query should return no results."""
        results = bm25_index.search("quantum computing neural networks")
        assert len(results) == 0

    def test_empty_index_search(self):
        """Search on empty index should return empty list."""
        index = BM25Index()
        index.build({})
        results = index.search("test query")
        assert results == []

    def test_tokenize_lowercases(self):
        """Tokenizer should lowercase all tokens."""
        tokens = BM25Index._tokenize("Hello World TEST")
        assert all(t == t.lower() for t in tokens)

    def test_tokenize_splits_on_punctuation(self):
        """Tokenizer should split on punctuation boundaries."""
        tokens = BM25Index._tokenize("error-code: E100")
        assert "error" in tokens
        assert "code" in tokens
        assert "e100" in tokens  # lowercased


class TestRRFFusion:
    """Tests for the Reciprocal Rank Fusion logic."""

    def test_rrf_combines_results(self):
        """RRF should combine results from both search sources."""
        bm25 = [("doc-A", 5.0), ("doc-B", 3.0), ("doc-C", 1.0)]
        vector = [("doc-B", 0.95), ("doc-D", 0.90), ("doc-A", 0.80)]

        fused = HybridSearcher._reciprocal_rank_fusion(
            bm25, vector, k=60, bm25_weight=0.4, vector_weight=0.6
        )

        fused_ids = [f[0] for f in fused]
        # All unique docs should appear
        assert set(fused_ids) == {"doc-A", "doc-B", "doc-C", "doc-D"}

    def test_rrf_boosts_docs_in_both_lists(self):
        """Documents appearing in both lists should rank higher."""
        bm25 = [("doc-A", 5.0), ("doc-B", 3.0)]
        vector = [("doc-A", 0.95), ("doc-C", 0.90)]

        fused = HybridSearcher._reciprocal_rank_fusion(
            bm25, vector, k=60, bm25_weight=0.5, vector_weight=0.5
        )

        # doc-A is in both, should be first
        assert fused[0][0] == "doc-A"

    def test_rrf_handles_empty_bm25(self):
        """RRF should work when BM25 returns no results."""
        bm25 = []
        vector = [("doc-A", 0.95), ("doc-B", 0.90)]

        fused = HybridSearcher._reciprocal_rank_fusion(
            bm25, vector, k=60, bm25_weight=0.4, vector_weight=0.6
        )

        assert len(fused) == 2

    def test_rrf_handles_empty_vector(self):
        """RRF should work when vector search returns no results."""
        bm25 = [("doc-A", 5.0), ("doc-B", 3.0)]
        vector = []

        fused = HybridSearcher._reciprocal_rank_fusion(
            bm25, vector, k=60, bm25_weight=0.4, vector_weight=0.6
        )

        assert len(fused) == 2


class TestWeightTable:
    """Tests for the dynamic weight table configuration."""

    def test_all_query_types_have_weights(self):
        """Every QueryType should have an entry in the weight table."""
        for qt in QueryType:
            assert qt in _WEIGHT_TABLE, f"Missing weight for {qt}"

    def test_weights_sum_to_one(self):
        """BM25 + vector weights should sum to 1.0 for each query type."""
        for qt, (w_bm25, w_vec) in _WEIGHT_TABLE.items():
            assert abs(w_bm25 + w_vec - 1.0) < 0.01, (
                f"Weights for {qt} sum to {w_bm25 + w_vec}, expected 1.0"
            )

    def test_troubleshoot_is_bm25_heavy(self):
        """Troubleshooting queries should favor BM25 for keyword matching."""
        w_bm25, w_vec = _WEIGHT_TABLE[QueryType.TROUBLESHOOT]
        assert w_bm25 > w_vec


class TestHybridSearchOffline:
    """Tests for the offline hybrid search (no Milvus dependency)."""

    def test_offline_search_returns_results(self, sample_chunks):
        """Offline search should return ranked results."""
        import numpy as np
        np.random.seed(42)

        searcher = HybridSearcher()
        results = searcher.search_offline(
            query="Where are the RAM slots?",
            query_vector=np.random.randn(384).tolist(),
            chunks=sample_chunks,
            query_type=QueryType.LOCATE_COMPONENT,
            top_k=3,
        )
        assert len(results) > 0

    def test_offline_search_has_rrf_scores(self, sample_chunks):
        """Offline results should have non-zero RRF scores."""
        import numpy as np
        np.random.seed(42)

        searcher = HybridSearcher()
        results = searcher.search_offline(
            query="RAM installation",
            query_vector=np.random.randn(384).tolist(),
            chunks=sample_chunks,
            top_k=3,
        )
        for r in results:
            assert r.rrf_score > 0

    def test_offline_search_respects_top_k(self, sample_chunks):
        """Should return at most top_k results."""
        import numpy as np
        np.random.seed(42)

        searcher = HybridSearcher()
        results = searcher.search_offline(
            query="motherboard",
            query_vector=np.random.randn(384).tolist(),
            chunks=sample_chunks,
            top_k=1,
        )
        assert len(results) <= 1

    def test_offline_search_empty_chunks(self):
        """Should handle empty chunks list gracefully."""
        searcher = HybridSearcher()
        results = searcher.search_offline(
            query="test",
            query_vector=[0.0] * 384,
            chunks=[],
            top_k=5,
        )
        assert results == []

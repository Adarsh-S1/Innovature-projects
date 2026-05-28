"""
Unit tests for ImageSearcher offline mode and composite scoring.

Tests cover:
- Offline image search with synthetic vectors
- Composite scoring formula (CLIP + caption + co-location + type bonus)
- Image type bonus weights
- Minimum score filtering
- top_k limiting
"""

import pytest
import numpy as np

from app.retrieval.image_searcher import (
    ImageSearcher,
    _IMAGE_TYPE_BONUS,
)


@pytest.fixture
def sample_images():
    """Sample image records with synthetic vectors for offline testing."""
    np.random.seed(42)
    dim_clip = 768
    dim_text = 384

    return [
        {
            "image_id": "img-1",
            "doc_id": "doc-1",
            "page_number": 3,
            "caption": "Motherboard RAM slot layout",
            "description": "Shows the location of RAM slots on the motherboard.",
            "topic_concept": "ram_slots",
            "keyword_tags": '["ram", "motherboard", "slot"]',
            "image_type": "schematic",
            "storage_url": "images/doc-1/img-1.png",
            "thumbnail_url": "thumbs/doc-1/img-1_thumb.png",
            "source_file": "manual.pdf",
            "product_id": "PROD-1",
            "linked_chunk_ids": '["c1", "c2"]',
            "clip_vector": np.random.randn(dim_clip).tolist(),
            "caption_vector": np.random.randn(dim_text).tolist(),
        },
        {
            "image_id": "img-2",
            "doc_id": "doc-1",
            "page_number": 5,
            "caption": "Power supply wiring",
            "description": "Wiring diagram for the PSU connections.",
            "topic_concept": "power_supply",
            "keyword_tags": '["power", "wiring", "psu"]',
            "image_type": "wiring_diagram",
            "storage_url": "images/doc-1/img-2.png",
            "thumbnail_url": "thumbs/doc-1/img-2_thumb.png",
            "source_file": "manual.pdf",
            "product_id": "PROD-1",
            "linked_chunk_ids": '["c3"]',
            "clip_vector": np.random.randn(dim_clip).tolist(),
            "caption_vector": np.random.randn(dim_text).tolist(),
        },
        {
            "image_id": "img-3",
            "doc_id": "doc-1",
            "page_number": 10,
            "caption": "Product photo exterior",
            "description": "Photograph of the product from the front.",
            "topic_concept": "exterior_view",
            "keyword_tags": '["photo", "product"]',
            "image_type": "photo",
            "storage_url": "images/doc-1/img-3.png",
            "thumbnail_url": "thumbs/doc-1/img-3_thumb.png",
            "source_file": "manual.pdf",
            "product_id": "PROD-1",
            "linked_chunk_ids": "[]",
            "clip_vector": np.random.randn(dim_clip).tolist(),
            "caption_vector": np.random.randn(dim_text).tolist(),
        },
    ]


class TestImageTypeBonus:
    """Tests for the image type relevance bonus weights."""

    def test_schematic_has_highest_bonus(self):
        """Schematics should have the highest type bonus."""
        assert _IMAGE_TYPE_BONUS["schematic"] == 1.0

    def test_photo_lower_than_schematic(self):
        """Photos should have lower bonus than schematics."""
        assert _IMAGE_TYPE_BONUS["photo"] < _IMAGE_TYPE_BONUS["schematic"]

    def test_all_types_have_bonus(self):
        """All ImageType values should have a defined bonus."""
        from app.models.domain import ImageType
        for img_type in ImageType:
            assert img_type.value in _IMAGE_TYPE_BONUS


class TestImageSearcherOffline:
    """Tests for the offline image search (no Milvus dependency)."""

    def test_offline_returns_results(self, sample_images):
        """Offline search should return scored results."""
        np.random.seed(42)
        searcher = ImageSearcher()
        results = searcher.search_offline(
            query_text_vector=np.random.randn(384).tolist(),
            query_clip_vector=np.random.randn(768).tolist(),
            image_records=sample_images,
            top_k=5,
            min_score=0.0,  # Accept everything for testing
        )
        assert len(results) > 0

    def test_offline_composite_score_range(self, sample_images):
        """Composite scores should be in a reasonable range."""
        np.random.seed(42)
        searcher = ImageSearcher()
        results = searcher.search_offline(
            query_text_vector=np.random.randn(384).tolist(),
            query_clip_vector=np.random.randn(768).tolist(),
            image_records=sample_images,
            top_k=5,
            min_score=0.0,
        )
        for r in results:
            # Type bonus alone guarantees a minimum positive score
            assert r.composite_score >= 0.0

    def test_offline_colocation_bonus(self, sample_images):
        """Images on relevant pages should get a co-location bonus."""
        np.random.seed(42)
        searcher = ImageSearcher()
        # Page 3 has img-1
        results = searcher.search_offline(
            query_text_vector=np.random.randn(384).tolist(),
            query_clip_vector=np.random.randn(768).tolist(),
            image_records=sample_images,
            relevant_page_numbers={3},
            top_k=5,
            min_score=0.0,
        )
        # Find img-1 (page 3)
        img1 = next((r for r in results if r.image_id == "img-1"), None)
        assert img1 is not None
        assert img1.colocation_bonus == 1.0

    def test_offline_no_colocation_different_page(self, sample_images):
        """Images NOT on relevant pages should get zero co-location bonus."""
        np.random.seed(42)
        searcher = ImageSearcher()
        results = searcher.search_offline(
            query_text_vector=np.random.randn(384).tolist(),
            query_clip_vector=np.random.randn(768).tolist(),
            image_records=sample_images,
            relevant_page_numbers={99},  # No images on page 99
            top_k=5,
            min_score=0.0,
        )
        for r in results:
            assert r.colocation_bonus == 0.0

    def test_offline_respects_top_k(self, sample_images):
        """Should return at most top_k results."""
        np.random.seed(42)
        searcher = ImageSearcher()
        results = searcher.search_offline(
            query_text_vector=np.random.randn(384).tolist(),
            query_clip_vector=np.random.randn(768).tolist(),
            image_records=sample_images,
            top_k=1,
            min_score=0.0,
        )
        assert len(results) <= 1

    def test_offline_filters_by_min_score(self, sample_images):
        """Results below min_score should be filtered out."""
        np.random.seed(42)
        searcher = ImageSearcher()
        results = searcher.search_offline(
            query_text_vector=np.random.randn(384).tolist(),
            query_clip_vector=np.random.randn(768).tolist(),
            image_records=sample_images,
            top_k=10,
            min_score=999.0,  # Impossible threshold
        )
        assert len(results) == 0

    def test_offline_empty_images(self):
        """Should handle empty image list gracefully."""
        searcher = ImageSearcher()
        results = searcher.search_offline(
            query_text_vector=[0.0] * 384,
            query_clip_vector=[0.0] * 768,
            image_records=[],
            top_k=5,
            min_score=0.0,
        )
        assert results == []

    def test_offline_results_sorted_by_score(self, sample_images):
        """Results should be sorted by composite_score descending."""
        np.random.seed(42)
        searcher = ImageSearcher()
        results = searcher.search_offline(
            query_text_vector=np.random.randn(384).tolist(),
            query_clip_vector=np.random.randn(768).tolist(),
            image_records=sample_images,
            top_k=5,
            min_score=0.0,
        )
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_offline_parses_json_keyword_tags(self, sample_images):
        """JSON string keyword_tags should be parsed into lists."""
        np.random.seed(42)
        searcher = ImageSearcher()
        results = searcher.search_offline(
            query_text_vector=np.random.randn(384).tolist(),
            query_clip_vector=np.random.randn(768).tolist(),
            image_records=sample_images,
            top_k=5,
            min_score=0.0,
        )
        for r in results:
            assert isinstance(r.keyword_tags, list)
            assert isinstance(r.linked_chunk_ids, list)

"""
End-to-end pipeline integration tests.

Tests the ingestion pipeline components in isolation (mocked external services)
and the data flow through the chunking + embedding + linking stages.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.ingestion.pdf_parser import (
    ExtractedPage,
    ExtractedImage,
    ParsedDocument,
    PDFParser,
)
from app.ingestion.chunker import SemanticChunker
from app.retrieval.cross_modal_linker import CrossModalLinker
from app.retrieval.reranker import Reranker
from app.retrieval.hybrid_searcher import SearchResult
from app.models.domain import TextChunk, ImageRecord, ChunkType, ImageType


class TestPDFParserEdgeCases:
    """Tests for PDF parser edge cases."""

    def test_nonexistent_file_raises(self):
        """Should raise PDFParsingError for missing files."""
        from app.core.exceptions import PDFParsingError
        parser = PDFParser()
        with pytest.raises(PDFParsingError):
            parser.parse("/nonexistent/path/test.pdf")

    def test_parse_from_bytes_invalid_pdf(self):
        """Should raise PDFParsingError for invalid PDF bytes."""
        from app.core.exceptions import PDFParsingError
        parser = PDFParser()
        with pytest.raises(PDFParsingError):
            parser.parse_from_bytes(b"not a real pdf", "fake.pdf")


class TestCrossModalLinker:
    """Tests for the cross-modal linking pipeline."""

    @pytest.fixture
    def sample_chunks_and_images(self):
        """Create sample chunks and images for linking tests."""
        chunks = [
            TextChunk(
                chunk_id="c1",
                doc_id="doc-1",
                page_start=1,
                page_end=1,
                text="The motherboard has four RAM slots. See Figure 1.",
                section_path=["Hardware"],
                chunk_type=ChunkType.PARAGRAPH,
            ),
            TextChunk(
                chunk_id="c2",
                doc_id="doc-1",
                page_start=3,
                page_end=3,
                text="Power supply connections are shown in the diagram.",
                section_path=["Hardware", "PSU"],
                chunk_type=ChunkType.PARAGRAPH,
            ),
        ]
        images = [
            ImageRecord(
                image_id="img-1",
                doc_id="doc-1",
                page_number=1,
                caption="RAM slot layout",
                image_type=ImageType.SCHEMATIC,
            ),
            ImageRecord(
                image_id="img-2",
                doc_id="doc-1",
                page_number=3,
                caption="PSU wiring diagram",
                image_type=ImageType.WIRING_DIAGRAM,
            ),
        ]
        return chunks, images

    def test_spatial_colocation_links(self, sample_chunks_and_images):
        """Chunks and images on the same page should be linked."""
        chunks, images = sample_chunks_and_images
        linker = CrossModalLinker(page_proximity=0)  # Same page only

        updated_chunks, updated_images = linker.link_chunks_and_images(
            chunks, images
        )

        # c1 (page 1) should link to img-1 (page 1)
        c1 = next(c for c in updated_chunks if c.chunk_id == "c1")
        assert "img-1" in c1.linked_images

        # c2 (page 3) should link to img-2 (page 3)
        c2 = next(c for c in updated_chunks if c.chunk_id == "c2")
        assert "img-2" in c2.linked_images

    def test_bidirectional_linking(self, sample_chunks_and_images):
        """Links should be bidirectional (chunk→image and image→chunk)."""
        chunks, images = sample_chunks_and_images
        linker = CrossModalLinker(page_proximity=0)

        _, updated_images = linker.link_chunks_and_images(chunks, images)

        # img-1 should link back to c1
        img1 = next(i for i in updated_images if i.image_id == "img-1")
        assert "c1" in img1.linked_chunk_ids

    def test_empty_input_returns_unchanged(self):
        """Empty inputs should return unchanged."""
        linker = CrossModalLinker()
        chunks, images = linker.link_chunks_and_images([], [])
        assert chunks == []
        assert images == []

    def test_proximity_range_links(self, sample_chunks_and_images):
        """With page_proximity=2, page 1 chunk can link to page 3 image."""
        chunks, images = sample_chunks_and_images
        linker = CrossModalLinker(page_proximity=2)

        updated_chunks, _ = linker.link_chunks_and_images(chunks, images)

        # c1 (page 1) with proximity=2 should reach img-2 (page 3)
        c1 = next(c for c in updated_chunks if c.chunk_id == "c1")
        assert "img-2" in c1.linked_images


class TestReranker:
    """Tests for the Reranker heuristic scoring."""

    @pytest.fixture
    def search_results(self):
        """Create sample SearchResult objects."""
        return [
            SearchResult(
                chunk_id="c1",
                doc_id="doc-1",
                text="RAM slots are located near the CPU socket on the motherboard.",
                page_start=1,
                page_end=1,
                section_path=["Hardware", "Motherboard"],
                source_file="manual.pdf",
                rrf_score=0.015,
            ),
            SearchResult(
                chunk_id="c2",
                doc_id="doc-1",
                text="The power supply unit provides stable voltage.",
                page_start=5,
                page_end=5,
                section_path=["Hardware"],
                source_file="manual.pdf",
                rrf_score=0.010,
            ),
            SearchResult(
                chunk_id="c3",
                doc_id="doc-1",
                text="Unrelated text about software installation procedures.",
                page_start=20,
                page_end=20,
                section_path=[],
                source_file="manual.pdf",
                rrf_score=0.002,
            ),
        ]

    def test_rerank_returns_results(self, search_results):
        """Reranker should return reranked results."""
        reranker = Reranker()
        results = reranker.rerank(
            query="Where are the RAM slots?",
            results=search_results,
            top_k=3,
            use_llm=False,
        )
        assert len(results) > 0

    def test_rerank_respects_top_k(self, search_results):
        """Should return at most top_k results."""
        reranker = Reranker()
        results = reranker.rerank(
            query="RAM slots",
            results=search_results,
            top_k=1,
            use_llm=False,
        )
        assert len(results) <= 1

    def test_rerank_sorted_by_final_score(self, search_results):
        """Results should be sorted by final_score descending."""
        reranker = Reranker()
        results = reranker.rerank(
            query="RAM motherboard",
            results=search_results,
            top_k=5,
            use_llm=False,
        )
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_input(self):
        """Should handle empty input gracefully."""
        reranker = Reranker()
        results = reranker.rerank(query="test", results=[], top_k=5)
        assert results == []

    def test_relevant_result_scores_higher(self, search_results):
        """Result with higher term overlap should score higher."""
        reranker = Reranker()
        results = reranker.rerank(
            query="RAM slots motherboard",
            results=search_results,
            top_k=5,
            use_llm=False,
        )
        if len(results) >= 2:
            # c1 (about RAM slots + motherboard) should rank above c3 (unrelated)
            result_ids = [r.chunk_id for r in results]
            if "c1" in result_ids and "c3" in result_ids:
                assert result_ids.index("c1") < result_ids.index("c3")


class TestIngestionPipelineUnit:
    """Unit tests for IngestionPipeline helper methods (no external deps)."""

    def test_chunker_to_linker_data_flow(self):
        """Test data flows correctly from chunker output to linker input."""
        doc = ParsedDocument(
            source_file="test.pdf",
            total_pages=1,
            pages=[
                ExtractedPage(
                    page_number=1,
                    text="The system has RAM slots for memory modules. "
                         "Installation requires opening the retention clips.",
                    headings=[{"level": "H1", "text": "Installation"}],
                    has_tables=False,
                ),
            ],
        )

        chunker = SemanticChunker()
        chunks = chunker.chunk_document(
            document=doc,
            doc_id="test-doc",
            product_id="SKU-1",
        )

        # Chunks should be valid TextChunk objects
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(c.doc_id == "test-doc" for c in chunks)

        # Linker should accept these chunks
        linker = CrossModalLinker()
        updated_chunks, _ = linker.link_chunks_and_images(chunks, [])
        assert len(updated_chunks) == len(chunks)

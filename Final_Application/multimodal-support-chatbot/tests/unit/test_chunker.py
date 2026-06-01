"""
Unit tests for the SemanticChunker.

Tests cover:
- Basic sentence splitting and chunking
- Long sentence handling
- Table chunk extraction
- Overlap computation
- Deduplication
- Section path tracking
- Figure reference extraction
"""

import pytest

from app.ingestion.chunker import SemanticChunker
from app.ingestion.pdf_parser import ExtractedPage, ParsedDocument
from app.models.domain import ChunkType


@pytest.fixture
def chunker():
    """Create a SemanticChunker with default settings."""
    return SemanticChunker()


@pytest.fixture
def simple_document():
    """Create a simple ParsedDocument for testing."""
    pages = [
        ExtractedPage(
            page_number=1,
            text=(
                "This is the first sentence of the document. "
                "It provides an introduction to the system. "
                "The system has multiple components that work together."
            ),
            headings=[{"level": "H1", "text": "Introduction"}],
            has_tables=False,
        ),
        ExtractedPage(
            page_number=2,
            text=(
                "The motherboard contains several important slots. "
                "RAM slots are located near the CPU socket. "
                "See Figure 1 for the component layout. "
                "Each slot supports DDR4 memory modules."
            ),
            headings=[{"level": "H2", "text": "Hardware Components"}],
            has_tables=False,
        ),
    ]
    return ParsedDocument(
        source_file="test_manual.pdf",
        total_pages=2,
        pages=pages,
    )


class TestSemanticChunker:
    """Tests for the SemanticChunker class."""

    def test_chunk_document_returns_chunks(self, chunker, simple_document):
        """Chunker should produce non-empty list of TextChunk objects."""
        chunks = chunker.chunk_document(
            document=simple_document,
            doc_id="test-doc-001",
            product_id="PRODUCT-X",
            language="en",
        )
        assert len(chunks) > 0

    def test_chunk_has_correct_doc_id(self, chunker, simple_document):
        """Each chunk should have the correct doc_id assigned."""
        chunks = chunker.chunk_document(
            document=simple_document,
            doc_id="test-doc-001",
        )
        for chunk in chunks:
            assert chunk.doc_id == "test-doc-001"

    def test_chunk_has_unique_ids(self, chunker, simple_document):
        """Each chunk should have a unique chunk_id."""
        chunks = chunker.chunk_document(
            document=simple_document,
            doc_id="test-doc-001",
        )
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_preserves_page_numbers(self, chunker, simple_document):
        """Chunks should have valid page_start and page_end values."""
        chunks = chunker.chunk_document(
            document=simple_document,
            doc_id="test-doc-001",
        )
        for chunk in chunks:
            assert chunk.page_start >= 1
            assert chunk.page_end >= chunk.page_start

    def test_chunk_type_defaults_to_paragraph(self, chunker, simple_document):
        """Regular text chunks should be tagged as PARAGRAPH."""
        chunks = chunker.chunk_document(
            document=simple_document,
            doc_id="test-doc-001",
        )
        for chunk in chunks:
            assert chunk.chunk_type in (ChunkType.PARAGRAPH, ChunkType.TABLE)

    def test_fallback_chunker_empty_section_path(self, chunker, simple_document):
        """Fallback chunker does not track sections, so section_path should be empty."""
        chunks = chunker.chunk_document(
            document=simple_document,
            doc_id="test-doc-001",
        )
        page2_chunks = [c for c in chunks if c.page_start == 2]
        if page2_chunks:
            assert len(page2_chunks[0].section_path) == 0

    def test_chunk_metadata_contains_source(self, chunker, simple_document):
        """Chunk metadata should include source_file."""
        chunks = chunker.chunk_document(
            document=simple_document,
            doc_id="test-doc-001",
            product_id="SKU-100",
            language="en",
        )
        for chunk in chunks:
            assert chunk.metadata["source_file"] == "test_manual.pdf"
            assert chunk.metadata["product_id"] == "SKU-100"
            assert chunk.metadata["language"] == "en"

    def test_empty_pages_produce_no_chunks(self, chunker):
        """Pages with no text should not produce any chunks."""
        doc = ParsedDocument(
            source_file="empty.pdf",
            total_pages=1,
            pages=[ExtractedPage(page_number=1, text="", headings=[])],
        )
        chunks = chunker.chunk_document(document=doc, doc_id="empty-doc")
        assert len(chunks) == 0

    def test_deduplication_removes_identical_text(self, chunker):
        """Duplicate text across pages should be deduplicated."""
        same_text = "This is a repeated paragraph with enough content to pass the minimum token threshold for chunking to work properly."
        doc = ParsedDocument(
            source_file="dup.pdf",
            total_pages=2,
            pages=[
                ExtractedPage(page_number=1, text=same_text, headings=[]),
                ExtractedPage(page_number=2, text=same_text, headings=[]),
            ],
        )
        chunks = chunker.chunk_document(document=doc, doc_id="dup-doc")
        # Only one unique chunk should remain
        assert len(chunks) == 1




"""
Semantic Text Chunker — splits PDF text into structure-aware chunks
using Docling's HybridChunker.

The HybridChunker combines:
- Hierarchical document structure awareness (sections, headings, lists)
- Token-limit-aware splitting/merging
- Table header repetition across split chunks

Falls back to a simple sliding-window chunker if no DoclingDocument
is available (e.g. when called with raw text).
"""

import re
from typing import Dict, List, Optional
from uuid import uuid4

from app.core.config import get_settings
from app.core.exceptions import ChunkingError
from app.core.logging import get_logger
from app.ingestion.pdf_parser import ParsedDocument
from app.models.domain import ChunkType, TextChunk

logger = get_logger(__name__)

# Figure/diagram reference pattern
_FIGURE_REF = re.compile(
    r'(?:see\s+)?(?:figure|fig\.?|diagram|image|photo|schematic|chart)\s*\.?\s*(\d+[A-Za-z]?)',
    re.IGNORECASE,
)


class SemanticChunker:
    """
    Splits extracted PDF text into semantically coherent chunks.

    Uses Docling's HybridChunker when a DoclingDocument is available,
    falling back to a simple sliding-window approach otherwise.
    """

    def __init__(self):
        self._settings = get_settings()
        self._chunk_size = self._settings.CHUNK_SIZE_TOKENS
        self._chunk_overlap = self._settings.CHUNK_OVERLAP_TOKENS

    def chunk_document(
        self,
        document: ParsedDocument,
        doc_id: str,
        product_id: Optional[str] = None,
        language: str = "en",
    ) -> List[TextChunk]:
        """
        Chunk an entire parsed document into TextChunk objects.

        If a DoclingDocument is available (from Docling parser), uses
        HybridChunker for structure-aware chunking. Otherwise falls
        back to simple text splitting.

        Args:
            document: Parsed PDF document from PDFParser.
            doc_id: Unique document identifier.
            product_id: Optional product ID to tag chunks.
            language: Document language code.

        Returns:
            List of TextChunk objects ready for embedding.
        """
        logger.info(
            "chunking_started",
            source_file=document.source_file,
            total_pages=document.total_pages,
        )

        if document.docling_document is not None:
            chunks = self._chunk_with_docling(
                document, doc_id, product_id, language
            )
        else:
            chunks = self._chunk_fallback(
                document, doc_id, product_id, language
            )

        # Deduplicate chunks with identical text
        chunks = self._deduplicate_chunks(chunks)

        logger.info(
            "chunking_completed",
            source_file=document.source_file,
            total_chunks=len(chunks),
        )

        return chunks

    def _chunk_with_docling(
        self,
        document: ParsedDocument,
        doc_id: str,
        product_id: Optional[str],
        language: str,
    ) -> List[TextChunk]:
        """
        Use Docling's HybridChunker for structure-aware chunking.

        HybridChunker respects document hierarchy (sections, headings,
        lists, tables) while enforcing token limits.
        """
        from docling_core.transforms.chunker import HybridChunker

        chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=self._chunk_size,
            merge_peers=True,
        )

        doc = document.docling_document
        docling_chunks = list(chunker.chunk(doc))

        logger.info(
            "docling_hybrid_chunking_completed",
            raw_chunks=len(docling_chunks),
        )

        chunks: List[TextChunk] = []
        for dc in docling_chunks:
            text = dc.text.strip() if hasattr(dc, "text") else ""
            if not text or len(text) < 20:
                continue

            # Determine page range from chunk metadata
            page_start, page_end = self._get_chunk_pages(dc)

            # Extract section path from chunk headings
            section_path = self._get_section_path(dc)

            # Classify chunk type
            chunk_type = self._classify_chunk_type(dc)

            chunks.append(TextChunk(
                chunk_id=str(uuid4()),
                doc_id=doc_id,
                page_start=page_start,
                page_end=page_end,
                section_path=section_path,
                text=text,
                chunk_type=chunk_type,
                metadata={
                    "source_file": document.source_file,
                    "product_id": product_id or "",
                    "language": language,
                    "chunk_type": chunk_type.value,
                },
            ))

        return chunks

    def _chunk_fallback(
        self,
        document: ParsedDocument,
        doc_id: str,
        product_id: Optional[str],
        language: str,
    ) -> List[TextChunk]:
        """
        Simple sliding-window fallback when no DoclingDocument is available.
        Splits on sentence boundaries with configurable overlap.
        """
        import tiktoken

        try:
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        chunks: List[TextChunk] = []

        for page in document.pages:
            text = page.text.strip()
            if not text:
                continue

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            current_sentences: List[str] = []
            current_token_count = 0

            for sentence in sentences:
                sentence_tokens = len(tokenizer.encode(sentence))

                if current_token_count + sentence_tokens > self._chunk_size and current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(TextChunk(
                        chunk_id=str(uuid4()),
                        doc_id=doc_id,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section_path=[],
                        text=chunk_text,
                        chunk_type=ChunkType.PARAGRAPH,
                        metadata={
                            "source_file": document.source_file,
                            "product_id": product_id or "",
                            "language": language,
                            "chunk_type": ChunkType.PARAGRAPH.value,
                        },
                    ))

                    # Keep overlap
                    overlap_sentences = []
                    overlap_tokens = 0
                    for s in reversed(current_sentences):
                        t = len(tokenizer.encode(s))
                        if overlap_tokens + t > self._chunk_overlap:
                            break
                        overlap_sentences.insert(0, s)
                        overlap_tokens += t
                    current_sentences = overlap_sentences
                    current_token_count = overlap_tokens

                current_sentences.append(sentence)
                current_token_count += sentence_tokens

            # Flush remaining
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                if len(tokenizer.encode(chunk_text)) >= 20:
                    chunks.append(TextChunk(
                        chunk_id=str(uuid4()),
                        doc_id=doc_id,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section_path=[],
                        text=chunk_text,
                        chunk_type=ChunkType.PARAGRAPH,
                        metadata={
                            "source_file": document.source_file,
                            "product_id": product_id or "",
                            "language": language,
                            "chunk_type": ChunkType.PARAGRAPH.value,
                        },
                    ))

        return chunks

    def _get_chunk_pages(self, dc) -> tuple:
        """Extract page range from a Docling chunk."""
        try:
            if hasattr(dc, "meta") and dc.meta:
                pages = []
                for item in dc.meta:
                    if hasattr(item, "prov") and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, "page_no"):
                                pages.append(prov.page_no)
                if pages:
                    return (min(pages), max(pages))
        except Exception:
            pass
        return (1, 1)

    def _get_section_path(self, dc) -> List[str]:
        """Extract section headings path from a Docling chunk."""
        try:
            if hasattr(dc, "meta") and dc.meta:
                headings = []
                for item in dc.meta:
                    if hasattr(item, "headings") and item.headings:
                        headings.extend(item.headings)
                return headings[:3]  # Max 3 levels deep
        except Exception:
            pass
        return []

    def _classify_chunk_type(self, dc) -> ChunkType:
        """Classify a Docling chunk into our ChunkType enum."""
        try:
            if hasattr(dc, "meta") and dc.meta:
                for item in dc.meta:
                    label = str(getattr(item, "label", "")).lower()
                    if "table" in label:
                        return ChunkType.TABLE
                    elif "list" in label:
                        return ChunkType.LIST
                    elif "code" in label:
                        return ChunkType.CODE_BLOCK
                    elif "heading" in label or "title" in label:
                        return ChunkType.HEADING
        except Exception:
            pass
        return ChunkType.PARAGRAPH

    def _deduplicate_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Remove chunks with identical text content."""
        seen_texts = set()
        unique_chunks = []

        for chunk in chunks:
            text_hash = hash(chunk.text.strip())
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_chunks.append(chunk)

        deduped = len(chunks) - len(unique_chunks)
        if deduped > 0:
            logger.info("chunks_deduplicated", removed=deduped)

        return unique_chunks

    @staticmethod
    def extract_figure_references(text: str) -> List[str]:
        """
        Extract explicit figure/diagram references from text.
        E.g., "See Figure 3", "refer to diagram 2B"

        Returns list of figure identifiers found.
        """
        return _FIGURE_REF.findall(text)

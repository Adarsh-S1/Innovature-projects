"""
Semantic Text Chunker — splits PDF text into overlapping chunks
with structural awareness.

Chunking Rules:
1. Respect section boundaries (H1/H2/H3 headings from PDF)
2. Sliding window: 512 tokens, 128-token overlap
3. Never split mid-sentence
4. Preserve tables as single atomic chunks
5. Tag each chunk with parent section + page range
"""

import re
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import tiktoken

from app.core.config import get_settings
from app.core.exceptions import ChunkingError
from app.core.logging import get_logger
from app.ingestion.pdf_parser import ExtractedPage, ParsedDocument
from app.models.domain import ChunkType, TextChunk

logger = get_logger(__name__)

# Sentence-ending pattern: period/question/exclamation followed by whitespace
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# Table pattern heuristics
_TABLE_PATTERN = re.compile(
    r'(?:^\s*\|.*\|.*$\n?){3,}',  # Markdown-style tables
    re.MULTILINE,
)

# Figure/diagram reference pattern
_FIGURE_REF = re.compile(
    r'(?:see\s+)?(?:figure|fig\.?|diagram|image|photo|schematic|chart)\s*\.?\s*(\d+[A-Za-z]?)',
    re.IGNORECASE,
)


class SemanticChunker:
    """
    Splits extracted PDF text into semantically coherent chunks
    with heading-aware boundaries and configurable overlap.
    """

    def __init__(self):
        self._settings = get_settings()
        self._chunk_size = self._settings.CHUNK_SIZE_TOKENS
        self._chunk_overlap = self._settings.CHUNK_OVERLAP_TOKENS
        try:
            self._tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_document(
        self,
        document: ParsedDocument,
        doc_id: str,
        product_id: Optional[str] = None,
        language: str = "en",
    ) -> List[TextChunk]:
        """
        Chunk an entire parsed document into TextChunk objects.

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

        # Build section path tracker from headings
        all_chunks: List[TextChunk] = []
        current_section_path: List[str] = []

        for page in document.pages:
            # Update section path based on headings found on this page
            current_section_path = self._update_section_path(
                current_section_path, page.headings
            )

            # If page has tables, extract them as atomic chunks
            if page.has_tables:
                table_chunks = self._extract_table_chunks(
                    page, doc_id, current_section_path, product_id, language, document.source_file
                )
                all_chunks.extend(table_chunks)

            # Chunk the regular text
            page_chunks = self._chunk_page_text(
                page, doc_id, current_section_path, product_id, language, document.source_file
            )
            all_chunks.extend(page_chunks)

        # Deduplicate chunks with identical text
        all_chunks = self._deduplicate_chunks(all_chunks)

        logger.info(
            "chunking_completed",
            source_file=document.source_file,
            total_chunks=len(all_chunks),
        )

        return all_chunks

    def _update_section_path(
        self, current_path: List[str], headings: List[Dict[str, str]]
    ) -> List[str]:
        """Update the section hierarchy path based on detected headings."""
        path = list(current_path)

        for heading in headings:
            level = heading["level"]
            text = heading["text"].strip()

            if level == "H1":
                path = [text]
            elif level == "H2":
                path = path[:1] + [text]
            elif level == "H3":
                path = path[:2] + [text]

        return path

    def _chunk_page_text(
        self,
        page: ExtractedPage,
        doc_id: str,
        section_path: List[str],
        product_id: Optional[str],
        language: str,
        source_file: str,
    ) -> List[TextChunk]:
        """
        Chunk text from a single page using sliding window
        with sentence-boundary awareness.
        """
        text = page.text.strip()
        if not text:
            return []

        # Split into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[TextChunk] = []
        current_tokens: List[str] = []
        current_sentences: List[str] = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = self._tokenizer.encode(sentence)
            sentence_token_count = len(sentence_tokens)

            # If a single sentence exceeds chunk size, split it by words
            if sentence_token_count > self._chunk_size:
                # Flush current buffer first
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(self._create_chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section_path=section_path,
                        product_id=product_id,
                        language=language,
                        source_file=source_file,
                        chunk_type=ChunkType.PARAGRAPH,
                    ))
                    current_sentences = []
                    current_token_count = 0

                # Split long sentence
                word_chunks = self._split_long_sentence(sentence)
                for wc in word_chunks:
                    chunks.append(self._create_chunk(
                        text=wc,
                        doc_id=doc_id,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section_path=section_path,
                        product_id=product_id,
                        language=language,
                        source_file=source_file,
                        chunk_type=ChunkType.PARAGRAPH,
                    ))
                continue

            # Check if adding this sentence would exceed the chunk size
            if current_token_count + sentence_token_count > self._chunk_size:
                # Emit current chunk
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(self._create_chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section_path=section_path,
                        product_id=product_id,
                        language=language,
                        source_file=source_file,
                        chunk_type=ChunkType.PARAGRAPH,
                    ))

                    # Compute overlap: keep last N tokens worth of sentences
                    overlap_sentences = self._compute_overlap_sentences(
                        current_sentences
                    )
                    current_sentences = overlap_sentences
                    current_token_count = sum(
                        len(self._tokenizer.encode(s)) for s in current_sentences
                    )

            current_sentences.append(sentence)
            current_token_count += sentence_token_count

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if len(self._tokenizer.encode(chunk_text)) >= 20:  # Skip very short tails
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    section_path=section_path,
                    product_id=product_id,
                    language=language,
                    source_file=source_file,
                    chunk_type=ChunkType.PARAGRAPH,
                ))

        return chunks

    def _extract_table_chunks(
        self,
        page: ExtractedPage,
        doc_id: str,
        section_path: List[str],
        product_id: Optional[str],
        language: str,
        source_file: str,
    ) -> List[TextChunk]:
        """
        Extract table-like content as atomic chunks.
        Tables are never split across chunk boundaries.
        """
        chunks = []
        text = page.text

        # Find table-like patterns (rows with consistent delimiters)
        table_matches = _TABLE_PATTERN.finditer(text)

        for match in table_matches:
            table_text = match.group(0).strip()
            if len(self._tokenizer.encode(table_text)) >= 10:
                chunks.append(self._create_chunk(
                    text=table_text,
                    doc_id=doc_id,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    section_path=section_path,
                    product_id=product_id,
                    language=language,
                    source_file=source_file,
                    chunk_type=ChunkType.TABLE,
                ))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving meaningful boundaries.
        Handles abbreviations, decimal numbers, etc.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return []

        # Split on sentence boundaries
        sentences = _SENTENCE_END.split(text)

        # Filter empty sentences and strip
        return [s.strip() for s in sentences if s.strip()]

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a sentence that exceeds max chunk size into word-level chunks."""
        words = sentence.split()
        chunks = []
        current_words = []
        current_count = 0

        for word in words:
            word_tokens = len(self._tokenizer.encode(word))
            if current_count + word_tokens > self._chunk_size and current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_count = 0
            current_words.append(word)
            current_count += word_tokens

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    def _compute_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        Compute overlap by keeping trailing sentences up to the overlap token count.
        """
        if not sentences:
            return []

        overlap_tokens = 0
        overlap_sentences = []

        for sentence in reversed(sentences):
            token_count = len(self._tokenizer.encode(sentence))
            if overlap_tokens + token_count > self._chunk_overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += token_count

        return overlap_sentences

    def _create_chunk(
        self,
        text: str,
        doc_id: str,
        page_start: int,
        page_end: int,
        section_path: List[str],
        product_id: Optional[str],
        language: str,
        source_file: str,
        chunk_type: ChunkType,
    ) -> TextChunk:
        """Create a TextChunk domain object."""
        return TextChunk(
            chunk_id=str(uuid4()),
            doc_id=doc_id,
            page_start=page_start,
            page_end=page_end,
            section_path=list(section_path),
            text=text,
            chunk_type=chunk_type,
            metadata={
                "source_file": source_file,
                "product_id": product_id or "",
                "language": language,
                "chunk_type": chunk_type.value,
            },
        )

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

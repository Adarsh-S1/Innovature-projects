"""
document_processor.py — PDF text extraction and recursive character chunking.

Extracts text from PDF files and splits them into overlapping chunks
with metadata (source document, page number, chunk index).

Uses LangChain's RecursiveCharacterTextSplitter for robust,
separator-aware chunking with proper overlap handling.
"""

import os
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def extract_text_from_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF file, page by page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts: [{"text": str, "page_number": int, "source": str}]
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            # Remove NUL bytes and other control characters that break PostgreSQL
            text = text.replace("\x00", "").strip()
            pages.append({
                "text": text,
                "page_number": i + 1,
                "source": pdf_path.name,
            })

    print(f"[DocProcessor] Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


# ─── LangChain Text Splitter ─────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    strip_whitespace=True,
    keep_separator=True,
)


def recursive_character_split(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Recursively split text into chunks using a hierarchy of separators.

    Delegates to LangChain's RecursiveCharacterTextSplitter for robust,
    separator-aware splitting with proper overlap at boundary positions.

    Strategy (separator hierarchy):
      1. Split by double newline (paragraphs)
      2. If chunks are still too large, split by single newline
      3. If still too large, split by sentence ('. ')
      4. If still too large, split by space
      5. Final fallback: split by character count

    Args:
        text: The text to split.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    # Use the module-level splitter if defaults match, otherwise create a new one
    if chunk_size == config.CHUNK_SIZE and chunk_overlap == config.CHUNK_OVERLAP:
        splitter = _splitter
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            strip_whitespace=True,
            keep_separator=True,
        )

    chunks = splitter.split_text(text)
    return [c for c in chunks if c.strip()]


def process_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Process a single PDF: extract text, chunk it, and attach metadata.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of chunk dicts: [{"content", "source_document", "page_number", "chunk_index"}]
    """
    pages = extract_text_from_pdf(pdf_path)
    all_chunks = []
    chunk_index = 0

    for page_data in pages:
        chunks = recursive_character_split(page_data["text"])
        for chunk_text in chunks:
            all_chunks.append({
                "content": chunk_text,
                "source_document": page_data["source"],
                "page_number": page_data["page_number"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    print(f"[DocProcessor] Created {len(all_chunks)} chunks from {Path(pdf_path).name}")
    return all_chunks


def process_all_documents(documents_dir: str | Path = config.DOCUMENTS_DIR) -> list[dict]:
    """
    Process all PDF files in the documents directory.

    Args:
        documents_dir: Path to the directory containing PDF files.

    Returns:
        List of all chunks from all documents.
    """
    documents_dir = Path(documents_dir)
    all_chunks = []

    pdf_files = sorted(documents_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[DocProcessor] No PDF files found in {documents_dir}")
        return []

    print(f"[DocProcessor] Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        chunks = process_pdf(pdf_path)
        all_chunks.extend(chunks)

    print(f"[DocProcessor] Total chunks across all documents: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # Quick test: process all documents
    chunks = process_all_documents()
    if chunks:
        print(f"\nSample chunk:")
        print(f"  Source: {chunks[0]['source_document']}")
        print(f"  Page: {chunks[0]['page_number']}")
        print(f"  Length: {len(chunks[0]['content'])} chars")
        print(f"  Content: {chunks[0]['content'][:200]}...")

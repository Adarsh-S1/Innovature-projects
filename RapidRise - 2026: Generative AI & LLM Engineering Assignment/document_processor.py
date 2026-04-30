"""
document_processor.py — PDF text extraction and recursive character chunking.

Extracts text from PDF files and splits them into overlapping chunks
with metadata (source document, page number, chunk index).
"""

import os
from pathlib import Path
from PyPDF2 import PdfReader
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


def recursive_character_split(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Recursively split text into chunks using a hierarchy of separators.

    Strategy:
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
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_recursive(text: str, sep_index: int = 0) -> list[str]:
        """Recursively split text using separators."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        if sep_index >= len(separators):
            # Final fallback: hard split by character count
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
            return chunks

        separator = separators[sep_index]

        if separator == "":
            # Hard character split
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
            return chunks

        # Split by current separator
        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for part in parts:
            # If adding this part exceeds chunk_size, save current and start new
            candidate = current_chunk + separator + part if current_chunk else part

            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # If the part itself is too large, recursively split it
                if len(part) > chunk_size:
                    sub_chunks = _split_recursive(part, sep_index + 1)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    raw_chunks = _split_recursive(text)

    # Apply overlap by creating sliding windows
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        overlapped = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev = raw_chunks[i - 1]
            curr = raw_chunks[i]
            # Take the last `overlap` characters from the previous chunk
            overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            # Prepend overlap if it doesn't create too large a chunk
            combined = overlap_text + " " + curr
            if len(combined) <= chunk_size * 1.5:
                overlapped.append(combined.strip())
            else:
                overlapped.append(curr)
        return overlapped

    return raw_chunks


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

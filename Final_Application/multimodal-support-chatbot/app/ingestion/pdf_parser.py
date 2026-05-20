"""
PDF Parser — extracts text and images from PDF files using PyMuPDF (fitz).

Handles:
- Raw text extraction per page with structural metadata
- Embedded image extraction per page with bounding boxes
- Heading/section structure detection (H1/H2/H3)
- Password-protected PDF detection
- Scanned PDF detection (empty text layer)
"""

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import PDFParsingError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedImage:
    """Raw image extracted from a PDF page."""

    image_bytes: bytes
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    width: int
    height: int
    image_index: int  # index on the page
    extension: str = "png"


@dataclass
class ExtractedPage:
    """All data extracted from a single PDF page."""

    page_number: int
    text: str
    images: List[ExtractedImage] = field(default_factory=list)
    headings: List[Dict[str, str]] = field(default_factory=list)  # [{"level": "H1", "text": "..."}]
    has_tables: bool = False


@dataclass
class ParsedDocument:
    """Complete extraction result from a PDF."""

    source_file: str
    total_pages: int
    pages: List[ExtractedPage]
    metadata: Dict[str, str] = field(default_factory=dict)


class PDFParser:
    """
    Extracts text and images from PDF files using PyMuPDF.

    Handles edge cases:
    - Password-protected PDFs → raises PDFParsingError
    - Scanned PDFs → flags pages with no text for potential OCR fallback
    - Large PDFs → page-by-page streaming extraction
    """

    def __init__(self):
        self._settings = get_settings()
        self._min_image_width = self._settings.MIN_IMAGE_WIDTH
        self._min_image_height = self._settings.MIN_IMAGE_HEIGHT

    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file and extract all text and images.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            ParsedDocument with text and images per page.

        Raises:
            PDFParsingError: If the PDF cannot be opened or is password-protected.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise PDFParsingError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise PDFParsingError(f"Failed to open PDF '{pdf_path}': {e}")

        # Check for password-protected PDFs
        if doc.is_encrypted:
            doc.close()
            raise PDFParsingError(
                f"PDF '{path.name}' is password-protected. "
                "Please provide an unencrypted version.",
                context={"source_file": path.name},
            )

        logger.info(
            "pdf_parsing_started",
            source_file=path.name,
            total_pages=doc.page_count,
        )

        pages: List[ExtractedPage] = []
        empty_text_pages = 0

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            extracted_page = self._extract_page(page, page_num + 1)  # 1-indexed
            pages.append(extracted_page)

            if not extracted_page.text.strip():
                empty_text_pages += 1

        doc.close()

        # Detect scanned PDFs (majority of pages have no text)
        total_pages = len(pages)
        if total_pages > 0 and (empty_text_pages / total_pages) > 0.5:
            logger.warning(
                "scanned_pdf_detected",
                source_file=path.name,
                empty_pages=empty_text_pages,
                total_pages=total_pages,
            )

        total_images = sum(len(p.images) for p in pages)
        logger.info(
            "pdf_parsing_completed",
            source_file=path.name,
            total_pages=len(pages),
            total_images=total_images,
            empty_text_pages=empty_text_pages,
        )

        return ParsedDocument(
            source_file=path.name,
            total_pages=len(pages),
            pages=pages,
            metadata={
                "format": doc.metadata.get("format", "") if hasattr(doc, "metadata") else "",
                "title": "",
                "author": "",
            },
        )

    def parse_from_bytes(self, pdf_bytes: bytes, filename: str) -> ParsedDocument:
        """
        Parse a PDF from in-memory bytes (e.g. from an upload).

        Args:
            pdf_bytes: Raw PDF file bytes.
            filename: Original filename for metadata.

        Returns:
            ParsedDocument with text and images per page.
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise PDFParsingError(f"Failed to open PDF '{filename}' from bytes: {e}")

        if doc.is_encrypted:
            doc.close()
            raise PDFParsingError(
                f"PDF '{filename}' is password-protected.",
                context={"source_file": filename},
            )

        logger.info(
            "pdf_parsing_started",
            source_file=filename,
            total_pages=doc.page_count,
        )

        pages: List[ExtractedPage] = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            extracted_page = self._extract_page(page, page_num + 1)
            pages.append(extracted_page)

        doc.close()

        total_images = sum(len(p.images) for p in pages)
        logger.info(
            "pdf_parsing_completed",
            source_file=filename,
            total_pages=len(pages),
            total_images=total_images,
        )

        return ParsedDocument(
            source_file=filename,
            total_pages=len(pages),
            pages=pages,
        )

    def _extract_page(self, page: fitz.Page, page_number: int) -> ExtractedPage:
        """Extract text, images, and structure from a single page."""
        # Extract text
        text = page.get_text("text")

        # Extract headings from text blocks with font-size heuristics
        headings = self._detect_headings(page)

        # Detect tables using pdfplumber-style heuristic (lines/rects)
        has_tables = self._detect_tables(page)

        # Extract images
        images = self._extract_images(page, page_number)

        return ExtractedPage(
            page_number=page_number,
            text=text,
            images=images,
            headings=headings,
            has_tables=has_tables,
        )

    def _detect_headings(self, page: fitz.Page) -> List[Dict[str, str]]:
        """
        Detect headings using font size analysis.
        Blocks with larger font sizes relative to the page's body text
        are classified as headings.
        """
        headings = []
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in blocks:
            if block.get("type") != 0:  # Only text blocks
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    font_size = span.get("size", 0)
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2**4)  # Bold flag

                    if not text or len(text) < 2:
                        continue

                    # Heuristic: headings are bold and/or have larger font size
                    if font_size >= 16 and is_bold:
                        headings.append({"level": "H1", "text": text})
                    elif font_size >= 14 and is_bold:
                        headings.append({"level": "H2", "text": text})
                    elif font_size >= 12 and is_bold and len(text) < 120:
                        headings.append({"level": "H3", "text": text})

        return headings

    def _detect_tables(self, page: fitz.Page) -> bool:
        """
        Heuristic table detection: look for grid-like arrangements
        of horizontal and vertical lines.
        """
        drawings = page.get_drawings()
        if not drawings:
            return False

        h_lines = 0
        v_lines = 0
        for d in drawings:
            for item in d.get("items", []):
                if item[0] == "l":  # line
                    p1, p2 = item[1], item[2]
                    dx = abs(p2.x - p1.x)
                    dy = abs(p2.y - p1.y)
                    if dx > 50 and dy < 5:
                        h_lines += 1
                    elif dy > 20 and dx < 5:
                        v_lines += 1

        # If we see a grid pattern, it's likely a table
        return h_lines >= 3 and v_lines >= 2

    def _extract_images(
        self, page: fitz.Page, page_number: int
    ) -> List[ExtractedImage]:
        """
        Extract embedded images from a page, filtering by minimum size
        and skipping decorative elements.
        """
        images = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = page.parent.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                width = base_image["width"]
                height = base_image["height"]
                ext = base_image.get("ext", "png")

                # Filter: minimum resolution
                if width < self._min_image_width or height < self._min_image_height:
                    continue

                # Filter: skip tiny decorative elements (icons, bullets)
                area = width * height
                if area < 15000:  # ~122x122 minimum useful area
                    continue

                # Filter: skip extreme aspect ratios (likely decorative bars)
                aspect_ratio = max(width, height) / max(min(width, height), 1)
                if aspect_ratio > 10:
                    continue

                # Try to get bounding box from page
                bbox = self._get_image_bbox(page, xref)

                images.append(
                    ExtractedImage(
                        image_bytes=image_bytes,
                        page_number=page_number,
                        bbox=bbox,
                        width=width,
                        height=height,
                        image_index=img_index,
                        extension=ext,
                    )
                )

            except Exception as e:
                logger.warning(
                    "image_extraction_failed",
                    page=page_number,
                    image_index=img_index,
                    error=str(e),
                )
                continue

        return images

    def _get_image_bbox(
        self, page: fitz.Page, xref: int
    ) -> Tuple[float, float, float, float]:
        """Try to get bounding box for an image on the page."""
        try:
            for img_block in page.get_image_info():
                if img_block.get("xref") == xref:
                    bbox = img_block.get("bbox", (0, 0, 0, 0))
                    return tuple(bbox)
        except Exception:
            pass
        return (0.0, 0.0, 0.0, 0.0)

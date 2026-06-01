"""
PDF Parser — extracts text and images from PDF files using Docling.

Handles:
- Rich text extraction per page with structural metadata
- Embedded image extraction per page with bounding boxes
- Heading/section structure detection (H1/H2/H3)
- Figure caption extraction from the PDF's own layout
- Scanned PDF detection (OCR-capable via Docling)
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    caption: str = ""  # Caption extracted from PDF layout (if available)


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
    docling_document: Optional[object] = None  # Raw DoclingDocument for chunker


class PDFParser:
    """
    Extracts text and images from PDF files using Docling.

    Docling provides AI-powered layout analysis, heading detection,
    table extraction, OCR, and figure-caption association — all
    running locally without external API calls.
    """

    def __init__(self):
        self._settings = get_settings()
        self._min_image_width = self._settings.MIN_IMAGE_WIDTH
        self._min_image_height = self._settings.MIN_IMAGE_HEIGHT
        self._converter = None

    def _ensure_converter(self):
        """Lazy-load Docling DocumentConverter."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = True

            self._converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
            logger.info("docling_converter_initialized")

    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file and extract all text and images.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            ParsedDocument with text and images per page.

        Raises:
            PDFParsingError: If the PDF cannot be opened.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise PDFParsingError(f"PDF file not found: {pdf_path}")

        self._ensure_converter()

        try:
            result = self._converter.convert(str(path))
        except Exception as e:
            raise PDFParsingError(f"Failed to parse PDF '{pdf_path}': {e}")

        return self._build_parsed_document(result, path.name)

    def parse_from_bytes(self, pdf_bytes: bytes, filename: str) -> ParsedDocument:
        """
        Parse a PDF from in-memory bytes (e.g. from an upload).

        Args:
            pdf_bytes: Raw PDF file bytes.
            filename: Original filename for metadata.

        Returns:
            ParsedDocument with text and images per page.
        """
        import tempfile
        import os

        self._ensure_converter()

        # Docling requires a file path, so write bytes to a temp file
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            result = self._converter.convert(tmp_path)
            return self._build_parsed_document(result, filename)
        except PDFParsingError:
            raise
        except Exception as e:
            raise PDFParsingError(f"Failed to parse PDF '{filename}' from bytes: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _build_parsed_document(self, result, source_name: str) -> ParsedDocument:
        """
        Convert Docling's ConversionResult into our ParsedDocument format.

        This extracts text, headings, tables, and images per page while
        preserving the raw DoclingDocument for use by the chunker.
        """
        from docling_core.types.doc.document import PictureItem, TableItem

        doc = result.document

        logger.info(
            "pdf_parsing_started",
            source_file=source_name,
        )

        # Build per-page data structures
        page_data: Dict[int, Dict] = {}

        # Extract text content per page from the document body
        for element, _level in doc.iterate_items():
            page_numbers = self._get_element_pages(element)
            text_content = element.text if hasattr(element, "text") else ""

            for page_num in page_numbers:
                if page_num not in page_data:
                    page_data[page_num] = {
                        "text_parts": [],
                        "headings": [],
                        "images": [],
                        "has_tables": False,
                    }

                if text_content:
                    page_data[page_num]["text_parts"].append(text_content)

                # Detect headings from Docling's structure
                if hasattr(element, "label") and "heading" in str(getattr(element, "label", "")).lower():
                    level = self._classify_heading_level(element)
                    if text_content.strip():
                        page_data[page_num]["headings"].append(
                            {"level": level, "text": text_content.strip()}
                        )

                # Detect tables
                if isinstance(element, TableItem):
                    page_data[page_num]["has_tables"] = True

        # Extract images with their captions
        img_global_index = 0
        for element, _level in doc.iterate_items():
            if isinstance(element, PictureItem):
                page_numbers = self._get_element_pages(element)
                page_num = page_numbers[0] if page_numbers else 1

                if page_num not in page_data:
                    page_data[page_num] = {
                        "text_parts": [],
                        "headings": [],
                        "images": [],
                        "has_tables": False,
                    }

                # Get caption from PDF layout (Docling auto-groups captions)
                caption = ""
                try:
                    caption = element.caption_text(doc=doc) or ""
                except Exception:
                    pass

                # Get image bytes
                image_bytes = self._extract_image_bytes(element)
                if image_bytes is None:
                    continue

                # Get image dimensions
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    width, height = img.size
                except Exception:
                    continue

                # Filter: minimum resolution
                if width < self._min_image_width or height < self._min_image_height:
                    continue

                # Filter: skip tiny decorative elements (icons, bullets)
                area = width * height
                if area < 10000:  # ~100x100 minimum useful area
                    continue

                # Filter: skip extreme aspect ratios (likely decorative bars)
                aspect_ratio = max(width, height) / max(min(width, height), 1)
                if aspect_ratio > 10:
                    continue

                # Get bounding box
                bbox = self._get_element_bbox(element)

                page_data[page_num]["images"].append(
                    ExtractedImage(
                        image_bytes=image_bytes,
                        page_number=page_num,
                        bbox=bbox,
                        width=width,
                        height=height,
                        image_index=img_global_index,
                        extension="png",
                        caption=caption,
                    )
                )
                img_global_index += 1

        # Build ExtractedPage objects
        if not page_data:
            # No structured content found — create a single empty page
            page_data[1] = {
                "text_parts": [],
                "headings": [],
                "images": [],
                "has_tables": False,
            }

        max_page = max(page_data.keys()) if page_data else 1
        pages: List[ExtractedPage] = []
        empty_text_pages = 0

        for page_num in range(1, max_page + 1):
            data = page_data.get(page_num, {
                "text_parts": [],
                "headings": [],
                "images": [],
                "has_tables": False,
            })

            page_text = "\n".join(data["text_parts"])
            if not page_text.strip():
                empty_text_pages += 1

            pages.append(ExtractedPage(
                page_number=page_num,
                text=page_text,
                images=data["images"],
                headings=data["headings"],
                has_tables=data["has_tables"],
            ))

        # Detect scanned PDFs (majority of pages have no text)
        total_pages = len(pages)
        if total_pages > 0 and (empty_text_pages / total_pages) > 0.5:
            logger.warning(
                "scanned_pdf_detected",
                source_file=source_name,
                empty_pages=empty_text_pages,
                total_pages=total_pages,
            )

        total_images = sum(len(p.images) for p in pages)
        logger.info(
            "pdf_parsing_completed",
            source_file=source_name,
            total_pages=total_pages,
            total_images=total_images,
            empty_text_pages=empty_text_pages,
        )

        # Capture metadata
        metadata = {}
        if hasattr(doc, "metadata"):
            metadata = {
                "title": getattr(doc.metadata, "title", "") or "",
                "author": getattr(doc.metadata, "author", "") or "",
            }

        return ParsedDocument(
            source_file=source_name,
            total_pages=total_pages,
            pages=pages,
            metadata=metadata,
            docling_document=doc,  # Preserve for HybridChunker
        )

    def _get_element_pages(self, element) -> List[int]:
        """Extract page numbers from a Docling document element."""
        try:
            if hasattr(element, "prov") and element.prov:
                pages = []
                for prov in element.prov:
                    if hasattr(prov, "page_no"):
                        pages.append(prov.page_no)
                return pages if pages else [1]
        except Exception:
            pass
        return [1]

    def _get_element_bbox(self, element) -> Tuple[float, float, float, float]:
        """Extract bounding box from a Docling document element."""
        try:
            if hasattr(element, "prov") and element.prov:
                prov = element.prov[0]
                if hasattr(prov, "bbox") and prov.bbox:
                    bbox = prov.bbox
                    return (bbox.l, bbox.t, bbox.r, bbox.b)
        except Exception:
            pass
        return (0.0, 0.0, 0.0, 0.0)

    def _classify_heading_level(self, element) -> str:
        """Classify a heading element into H1/H2/H3 based on Docling's label."""
        label = str(getattr(element, "label", "")).lower()
        if "section_header" in label or "title" in label:
            # Docling uses nesting level for heading hierarchy
            level = getattr(element, "level", 1)
            if isinstance(level, int):
                if level <= 1:
                    return "H1"
                elif level == 2:
                    return "H2"
                else:
                    return "H3"
        return "H3"

    def _extract_image_bytes(self, element) -> Optional[bytes]:
        """Extract raw image bytes from a Docling PictureItem."""
        try:
            img = element.get_image(doc=None)
            if img is None and hasattr(element, "image"):
                img = element.image
            if img is None:
                return None

            # If it's a PIL Image, convert to bytes
            if isinstance(img, Image.Image):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            # If it's bytes already
            if isinstance(img, bytes):
                return img

        except Exception as e:
            logger.debug("image_extraction_failed", error=str(e))

        return None

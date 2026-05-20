"""
Image Processor — handles filtering, classification, caption generation (GPT-4o Vision),
CLIP embedding, thumbnail generation, and perceptual hash deduplication.
"""

import base64
import hashlib
import io
import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import ImageProcessingError
from app.core.logging import get_logger
from app.ingestion.pdf_parser import ExtractedImage
from app.models.domain import ImageRecord, ImageType

logger = get_logger(__name__)

# Caption generation system prompt for GPT-4o Vision
_CAPTION_SYSTEM_PROMPT = """You are a technical documentation assistant. Given a diagram or image \
from a technical manual, provide:
1. A concise caption (max 20 words)
2. A detailed description (2-3 sentences covering components and function)
3. 5-10 technical keyword tags
4. The main topic concept this image explains
5. The image type classification (one of: schematic, wiring_diagram, chart, photo, table, flowchart, other)

Respond ONLY in valid JSON with this exact structure:
{
  "caption": "...",
  "description": "...",
  "keyword_tags": ["tag1", "tag2", ...],
  "topic_concept": "...",
  "image_type": "..."
}"""


class ImageProcessor:
    """
    Processes images extracted from PDFs:
    - Filters by size and aspect ratio
    - Generates captions via GPT-4o Vision
    - Generates CLIP embeddings
    - Creates thumbnails
    - Deduplicates via perceptual hashing
    """

    def __init__(self):
        self._settings = get_settings()
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_device = None
        self._openai_client = None
        self._seen_hashes: set = set()

    def _ensure_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self._settings.OPENAI_API_KEY,
                max_retries=self._settings.OPENAI_MAX_RETRIES,
                timeout=self._settings.OPENAI_TIMEOUT,
            )

    def _ensure_clip_model(self):
        """Lazy-load CLIP model and preprocessing."""
        if self._clip_model is None:
            try:
                import torch
                import clip as clip_module

                device = self._settings.CLIP_DEVICE
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                    logger.warning("clip_cuda_unavailable_falling_back_to_cpu")

                self._clip_device = device
                self._clip_model, self._clip_preprocess = clip_module.load(
                    self._settings.CLIP_MODEL_NAME, device=device
                )
                logger.info(
                    "clip_model_loaded",
                    model=self._settings.CLIP_MODEL_NAME,
                    device=device,
                )
            except ImportError:
                logger.warning("clip_not_installed_using_dummy_embeddings")
                self._clip_model = "dummy"
            except Exception as e:
                logger.error("clip_model_load_failed", error=str(e))
                self._clip_model = "dummy"

    def process_images(
        self,
        extracted_images: List[ExtractedImage],
        doc_id: str,
        source_file: str,
        product_id: Optional[str] = None,
    ) -> List[ImageRecord]:
        """
        Process a list of extracted images through the full pipeline.

        Args:
            extracted_images: Raw images from PDFParser.
            doc_id: Document identifier.
            source_file: Source PDF filename.
            product_id: Optional product ID.

        Returns:
            List of ImageRecord objects with captions, embeddings, and metadata.
        """
        logger.info(
            "image_processing_started",
            count=len(extracted_images),
            doc_id=doc_id,
        )

        records: List[ImageRecord] = []
        self._seen_hashes.clear()

        for img in extracted_images:
            try:
                record = self._process_single_image(
                    img, doc_id, source_file, product_id
                )
                if record is not None:
                    records.append(record)
            except Exception as e:
                logger.warning(
                    "image_processing_failed",
                    page=img.page_number,
                    index=img.image_index,
                    error=str(e),
                )
                continue

        logger.info(
            "image_processing_completed",
            processed=len(records),
            total=len(extracted_images),
        )

        return records

    def _process_single_image(
        self,
        img: ExtractedImage,
        doc_id: str,
        source_file: str,
        product_id: Optional[str],
    ) -> Optional[ImageRecord]:
        """Process a single image through filtering, captioning, and embedding."""
        # Step 1: Deduplication via perceptual hash
        img_hash = self._compute_hash(img.image_bytes)
        if img_hash in self._seen_hashes:
            logger.debug(
                "duplicate_image_skipped",
                page=img.page_number,
                hash=img_hash[:12],
            )
            return None
        self._seen_hashes.add(img_hash)

        # Step 2: Generate caption via GPT-4o Vision
        caption_data = self._generate_caption(img.image_bytes, img.extension)

        # Step 3: Generate CLIP embedding
        clip_vector = self._generate_clip_embedding(img.image_bytes)

        # Step 4: Create thumbnail
        thumbnail_bytes = self._create_thumbnail(img.image_bytes)

        # Step 5: Classify image type
        image_type = self._classify_image_type(
            caption_data.get("image_type", "other")
        )

        image_id = str(uuid4())

        return ImageRecord(
            image_id=image_id,
            doc_id=doc_id,
            page_number=img.page_number,
            caption=caption_data.get("caption", ""),
            description=caption_data.get("description", ""),
            topic_concept=caption_data.get("topic_concept", ""),
            keyword_tags=caption_data.get("keyword_tags", []),
            image_type=image_type,
            clip_vector=clip_vector,
            metadata={
                "source_file": source_file,
                "product_id": product_id or "",
                "bbox": list(img.bbox),
                "image_width": img.width,
                "image_height": img.height,
                "image_hash": img_hash,
                "raw_image_bytes": img.image_bytes,
                "thumbnail_bytes": thumbnail_bytes,
                "extension": img.extension,
            },
        )

    def _compute_hash(self, image_bytes: bytes) -> str:
        """Compute a perceptual hash for image deduplication."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # Resize to 8x8 and convert to grayscale for perceptual hash
            img_small = img.resize((8, 8), Image.Resampling.LANCZOS).convert("L")
            pixels = list(img_small.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if p > avg else "0" for p in pixels)
            return hashlib.md5(bits.encode()).hexdigest()
        except Exception:
            # Fallback: content hash
            return hashlib.md5(image_bytes).hexdigest()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _generate_caption(
        self, image_bytes: bytes, extension: str = "png"
    ) -> Dict[str, Any]:
        """
        Generate caption, description, tags using GPT-4o Vision.
        Falls back to empty metadata if the API is unavailable.
        """
        self._ensure_openai_client()

        try:
            # Encode image to base64
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = f"image/{extension}" if extension != "jpg" else "image/jpeg"

            response = self._openai_client.chat.completions.create(
                model=self._settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _CAPTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64_image}",
                                    "detail": "low",
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyze this technical image and provide the JSON response.",
                            },
                        ],
                    },
                ],
                max_tokens=500,
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                content = content.rsplit("```", 1)[0]

            result = json.loads(content)
            logger.debug("caption_generated", caption=result.get("caption", "")[:50])
            return result

        except json.JSONDecodeError as e:
            logger.warning("caption_json_parse_failed", error=str(e))
            return self._fallback_caption()
        except Exception as e:
            logger.warning("caption_generation_failed", error=str(e))
            return self._fallback_caption()

    def _fallback_caption(self) -> Dict[str, Any]:
        """Return fallback caption metadata when GPT-4o is unavailable."""
        return {
            "caption": "Technical image from manual",
            "description": "Image extracted from technical documentation.",
            "keyword_tags": ["technical", "manual"],
            "topic_concept": "technical_documentation",
            "image_type": "other",
        }

    def _generate_clip_embedding(self, image_bytes: bytes) -> List[float]:
        """Generate CLIP embedding for an image."""
        self._ensure_clip_model()

        if self._clip_model == "dummy":
            # Return dummy embedding for testing without CLIP
            return [0.0] * self._settings.CLIP_EMBEDDING_DIMENSIONS

        try:
            import torch

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_tensor = self._clip_preprocess(img).unsqueeze(0).to(self._clip_device)

            with torch.no_grad():
                features = self._clip_model.encode_image(image_tensor)
                # Normalize
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().flatten().tolist()

        except Exception as e:
            logger.warning("clip_embedding_failed", error=str(e))
            return [0.0] * self._settings.CLIP_EMBEDDING_DIMENSIONS

    def _create_thumbnail(
        self, image_bytes: bytes, max_size: Tuple[int, int] = (200, 200)
    ) -> bytes:
        """Create a thumbnail from image bytes."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.warning("thumbnail_creation_failed", error=str(e))
            return b""

    def _classify_image_type(self, raw_type: str) -> ImageType:
        """Map GPT-4o's classification string to the ImageType enum."""
        mapping = {
            "schematic": ImageType.SCHEMATIC,
            "wiring_diagram": ImageType.WIRING_DIAGRAM,
            "wiring diagram": ImageType.WIRING_DIAGRAM,
            "chart": ImageType.CHART,
            "photo": ImageType.PHOTO,
            "photograph": ImageType.PHOTO,
            "table": ImageType.TABLE,
            "flowchart": ImageType.FLOWCHART,
            "flow chart": ImageType.FLOWCHART,
            "diagram": ImageType.SCHEMATIC,
        }
        return mapping.get(raw_type.lower().strip(), ImageType.OTHER)

    def generate_clip_text_embedding(self, text: str) -> List[float]:
        """
        Generate CLIP text embedding for cross-modal search.
        Used to encode queries in the image search space.
        """
        self._ensure_clip_model()

        if self._clip_model == "dummy":
            return [0.0] * self._settings.CLIP_EMBEDDING_DIMENSIONS

        try:
            import torch
            import clip as clip_module

            tokens = clip_module.tokenize([text]).to(self._clip_device)
            with torch.no_grad():
                features = self._clip_model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().flatten().tolist()

        except Exception as e:
            logger.warning("clip_text_embedding_failed", error=str(e))
            return [0.0] * self._settings.CLIP_EMBEDDING_DIMENSIONS

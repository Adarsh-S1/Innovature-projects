"""
Image Searcher — cross-modal image retrieval using CLIP + caption vectors.

Retrieves relevant images from the Milvus image collection by combining
four scoring signals into a composite relevance score:

    Final Image Score =
        0.35 × CLIP_similarity(query_clip_vector, image_clip_vector)
      + 0.35 × cosine_sim(query_text_vector, caption_text_vector)
      + 0.20 × co_location_bonus (1 if on same page as a top text chunk)
      + 0.10 × image_type_relevance_bonus (schematic/diagram > photo)

Images are delivered via MinIO presigned URLs, not base64.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from app.core.config import get_settings
from app.core.exceptions import ImageRetrievalError
from app.core.logging import get_logger
from app.models.domain import ImageType

logger = get_logger(__name__)

# Image type relevance bonuses — schematics and diagrams are more useful
# in technical support contexts than photos or generic images.
_IMAGE_TYPE_BONUS: Dict[str, float] = {
    ImageType.SCHEMATIC.value: 1.0,
    ImageType.WIRING_DIAGRAM.value: 1.0,
    ImageType.FLOWCHART.value: 0.9,
    ImageType.CHART.value: 0.8,
    ImageType.TABLE.value: 0.7,
    ImageType.PHOTO.value: 0.5,
    ImageType.OTHER.value: 0.3,
}


@dataclass
class ImageSearchResult:
    """A single scored image result from the retrieval pipeline."""

    image_id: str
    doc_id: str = ""
    page_number: int = 0
    caption: str = ""
    description: str = ""
    topic_concept: str = ""
    keyword_tags: List[str] = field(default_factory=list)
    image_type: str = "other"
    storage_url: str = ""
    thumbnail_url: str = ""
    source_file: str = ""
    product_id: str = ""
    linked_chunk_ids: List[str] = field(default_factory=list)
    # Scoring
    clip_score: float = 0.0
    caption_score: float = 0.0
    colocation_bonus: float = 0.0
    type_bonus: float = 0.0
    composite_score: float = 0.0


class ImageSearcher:
    """
    Cross-modal image retrieval combining CLIP visual similarity,
    caption semantic similarity, spatial co-location, and image type priors.
    """

    def __init__(self):
        self._settings = get_settings()

    def search(
        self,
        query_text_vector: Optional[List[float]] = None,
        query_clip_vector: Optional[List[float]] = None,
        relevant_page_numbers: Optional[Set[int]] = None,
        top_k: Optional[int] = None,
        doc_id_filter: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[ImageSearchResult]:
        """
        Search for relevant images using multi-signal scoring.

        Args:
            query_text_vector: 3072-dim text embedding of the query.
            query_clip_vector: 768-dim CLIP text embedding of the query.
            relevant_page_numbers: Pages where top text results were found
                                   (used for co-location bonus).
            top_k: Max images to return.
            doc_id_filter: Restrict to a specific document.
            min_score: Minimum composite score to include.

        Returns:
            Sorted list of ImageSearchResult, descending by composite score.
        """
        top_k = top_k or self._settings.IMAGE_RETURN_TOP_K
        min_score = min_score or self._settings.CLIP_SIMILARITY_THRESHOLD
        relevant_pages = relevant_page_numbers or set()

        logger.info(
            "image_search_started",
            has_clip=query_clip_vector is not None,
            has_text=query_text_vector is not None,
            relevant_pages=len(relevant_pages),
            top_k=top_k,
        )

        candidates: Dict[str, Dict[str, Any]] = {}

        # ── Signal 1: CLIP vector search ─────────────────────────────────
        if query_clip_vector:
            clip_hits = self._clip_vector_search(
                query_clip_vector,
                top_k=self._settings.IMAGE_SEARCH_TOP_K,
                doc_id_filter=doc_id_filter,
            )
            for hit in clip_hits:
                iid = hit["image_id"]
                if iid not in candidates:
                    candidates[iid] = hit
                candidates[iid]["clip_distance"] = hit.get("clip_distance", 0.0)

        # ── Signal 2: Caption vector search ──────────────────────────────
        if query_text_vector:
            caption_hits = self._caption_vector_search(
                query_text_vector,
                top_k=self._settings.IMAGE_SEARCH_TOP_K,
                doc_id_filter=doc_id_filter,
            )
            for hit in caption_hits:
                iid = hit["image_id"]
                if iid not in candidates:
                    candidates[iid] = hit
                candidates[iid]["caption_distance"] = hit.get("caption_distance", 0.0)

        if not candidates:
            logger.info("image_search_no_candidates")
            return []

        # ── Composite scoring ────────────────────────────────────────────
        scored_results: List[ImageSearchResult] = []

        for iid, meta in candidates.items():
            clip_sim = meta.get("clip_distance", 0.0)
            caption_sim = meta.get("caption_distance", 0.0)

            # Co-location bonus: 1.0 if image is on a relevant page
            page_num = meta.get("page_number", -1)
            colocation = 1.0 if page_num in relevant_pages else 0.0

            # Image type bonus
            img_type = meta.get("image_type", "other")
            type_bonus = _IMAGE_TYPE_BONUS.get(img_type, 0.3)

            # Weighted composite (from Part 3 spec)
            composite = (
                0.35 * clip_sim
                + 0.35 * caption_sim
                + 0.20 * colocation
                + 0.10 * type_bonus
            )

            # keyword_tags is still VARCHAR (JSON-serialized), parse it
            from app.core.shared import parse_json_field
            keyword_tags = parse_json_field(meta.get("keyword_tags", "[]"))
            # linked_chunk_ids is now a native ARRAY field — no parsing needed
            linked_chunk_ids = meta.get("linked_chunk_ids", [])

            scored_results.append(ImageSearchResult(
                image_id=iid,
                doc_id=meta.get("doc_id", ""),
                page_number=page_num,
                caption=meta.get("caption", ""),
                description=meta.get("description", ""),
                topic_concept=meta.get("topic_concept", ""),
                keyword_tags=keyword_tags,
                image_type=img_type,
                storage_url=meta.get("storage_url", ""),
                thumbnail_url=meta.get("thumbnail_url", ""),
                source_file=meta.get("source_file", ""),
                product_id=meta.get("product_id", ""),
                linked_chunk_ids=linked_chunk_ids,
                clip_score=clip_sim,
                caption_score=caption_sim,
                colocation_bonus=colocation,
                type_bonus=type_bonus,
                composite_score=composite,
            ))

        # Filter by minimum score and sort
        scored_results = [r for r in scored_results if r.composite_score >= min_score]
        scored_results.sort(key=lambda r: r.composite_score, reverse=True)

        final = scored_results[:top_k]

        logger.info(
            "image_search_completed",
            total_candidates=len(candidates),
            above_threshold=len(scored_results),
            returned=len(final),
        )

        return final

    def _clip_vector_search(
        self,
        query_clip_vector: List[float],
        top_k: int = 10,
        doc_id_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search Milvus image collection by CLIP vector similarity."""
        from app.db.milvus_client import milvus_manager

        collection = milvus_manager.image_collection
        if collection is None:
            logger.warning("milvus_image_collection_unavailable")
            return []

        try:
            collection.load()

            from app.core.sanitize import build_milvus_filter
            expr = build_milvus_filter(doc_id=doc_id_filter)

            results = collection.search(
                data=[query_clip_vector],
                anns_field="clip_vector",
                param={
                    "metric_type": "COSINE",
                    "params": {"ef": 128},
                },
                limit=top_k,
                expr=expr,
                output_fields=[
                    "image_id", "doc_id", "page_number", "caption",
                    "description", "topic_concept", "keyword_tags",
                    "image_type", "storage_url", "thumbnail_url",
                    "linked_chunk_ids", "source_file", "product_id",
                ],
            )

            hits = []
            if results and len(results) > 0:
                for hit in results[0]:
                    data = {field: hit.entity.get(field) for field in [
                        "image_id", "doc_id", "page_number", "caption",
                        "description", "topic_concept", "keyword_tags",
                        "image_type", "storage_url", "thumbnail_url",
                        "linked_chunk_ids", "source_file", "product_id",
                    ]}
                    data["clip_distance"] = hit.distance
                    hits.append(data)

            return hits

        except Exception as e:
            logger.error("clip_vector_search_failed", error=str(e))
            return []

    def _caption_vector_search(
        self,
        query_text_vector: List[float],
        top_k: int = 10,
        doc_id_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search Milvus image collection by caption text vector similarity."""
        from app.db.milvus_client import milvus_manager

        collection = milvus_manager.image_collection
        if collection is None:
            logger.warning("milvus_image_collection_unavailable")
            return []

        try:
            collection.load()

            from app.core.sanitize import build_milvus_filter
            expr = build_milvus_filter(doc_id=doc_id_filter)

            results = collection.search(
                data=[query_text_vector],
                anns_field="caption_vector",
                param={
                    "metric_type": "COSINE",
                    "params": {"ef": 128},
                },
                limit=top_k,
                expr=expr,
                output_fields=[
                    "image_id", "doc_id", "page_number", "caption",
                    "description", "topic_concept", "keyword_tags",
                    "image_type", "storage_url", "thumbnail_url",
                    "linked_chunk_ids", "source_file", "product_id",
                ],
            )

            hits = []
            if results and len(results) > 0:
                for hit in results[0]:
                    data = {field: hit.entity.get(field) for field in [
                        "image_id", "doc_id", "page_number", "caption",
                        "description", "topic_concept", "keyword_tags",
                        "image_type", "storage_url", "thumbnail_url",
                        "linked_chunk_ids", "source_file", "product_id",
                    ]}
                    data["caption_distance"] = hit.distance
                    hits.append(data)

            return hits

        except Exception as e:
            logger.error("caption_vector_search_failed", error=str(e))
            return []

    def search_offline(
        self,
        query_text_vector: Optional[List[float]],
        query_clip_vector: Optional[List[float]],
        image_records: List[Dict[str, Any]],
        relevant_page_numbers: Optional[Set[int]] = None,
        top_k: int = 3,
        min_score: float = 0.1,
    ) -> List[ImageSearchResult]:
        """
        Offline image search — operates entirely in-memory without Milvus.
        Useful for testing.

        Args:
            query_text_vector: Query text embedding.
            query_clip_vector: Query CLIP embedding.
            image_records: List of dicts with image fields.
            relevant_page_numbers: Pages with top text results.
            top_k: Max results.
            min_score: Min composite score threshold.
        """
        relevant_pages = relevant_page_numbers or set()
        scored_results: List[ImageSearchResult] = []

        for img in image_records:
            clip_sim = 0.0
            caption_sim = 0.0

            # CLIP similarity
            if query_clip_vector and img.get("clip_vector"):
                qv = np.array(query_clip_vector, dtype=np.float32)
                iv = np.array(img["clip_vector"], dtype=np.float32)
                norms = np.linalg.norm(qv) * np.linalg.norm(iv)
                if norms > 0:
                    clip_sim = float(np.dot(qv, iv) / norms)

            # Caption similarity
            if query_text_vector and img.get("caption_vector"):
                qv = np.array(query_text_vector, dtype=np.float32)
                cv = np.array(img["caption_vector"], dtype=np.float32)
                norms = np.linalg.norm(qv) * np.linalg.norm(cv)
                if norms > 0:
                    caption_sim = float(np.dot(qv, cv) / norms)

            page_num = img.get("page_number", -1)
            colocation = 1.0 if page_num in relevant_pages else 0.0

            img_type = img.get("image_type", "other")
            type_bonus = _IMAGE_TYPE_BONUS.get(img_type, 0.3)

            composite = (
                0.35 * clip_sim
                + 0.35 * caption_sim
                + 0.20 * colocation
                + 0.10 * type_bonus
            )

            from app.core.shared import parse_json_field
            keyword_tags = parse_json_field(img.get("keyword_tags", []))
            linked_chunk_ids = parse_json_field(img.get("linked_chunk_ids", []))

            scored_results.append(ImageSearchResult(
                image_id=img.get("image_id", ""),
                doc_id=img.get("doc_id", ""),
                page_number=page_num,
                caption=img.get("caption", ""),
                description=img.get("description", ""),
                topic_concept=img.get("topic_concept", ""),
                keyword_tags=keyword_tags,
                image_type=img_type,
                storage_url=img.get("storage_url", ""),
                thumbnail_url=img.get("thumbnail_url", ""),
                source_file=img.get("source_file", ""),
                product_id=img.get("product_id", ""),
                linked_chunk_ids=linked_chunk_ids,
                clip_score=clip_sim,
                caption_score=caption_sim,
                colocation_bonus=colocation,
                type_bonus=type_bonus,
                composite_score=composite,
            ))

        scored_results = [r for r in scored_results if r.composite_score >= min_score]
        scored_results.sort(key=lambda r: r.composite_score, reverse=True)

        return scored_results[:top_k]

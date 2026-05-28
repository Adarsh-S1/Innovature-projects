"""
Agent 3: Visual Specialist — Cross-modal image retrieval via CLIP + caption vectors.

Operates ONLY when needs_image=True. Implements the 3-pass retrieval
strategy from Part 3 spec:
  Pass 1: Direct CLIP query (query text → clip_vector)
  Pass 2: Caption similarity (query text → caption_vector)
  Pass 3: Co-location resolution from linked_image_ids in retrieved chunks
"""

from typing import Any, Dict, List, Set

from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def visual_specialist(state: AgentState) -> dict:
    """
    Agent 3: Cross-modal image retrieval using CLIP and caption embeddings.

    Skipped entirely if needs_image=False. Uses the ImageSearcher from Part 3
    with 4-signal composite scoring.
    """
    needs_image = state.get("needs_image", False)

    if not needs_image:
        logger.info("visual_specialist_skipped", reason="needs_image=False")
        return {"retrieved_images": []}

    query = state.get("reformulated_query") or state.get("user_query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    logger.info(
        "visual_specialist_invoked",
        query_len=len(query),
        num_chunks=len(retrieved_chunks),
    )

    settings = get_settings()

    # ── Collect relevant page numbers from top text results (for co-location) ──
    relevant_pages: Set[int] = set()
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            relevant_pages.add(chunk.get("page_start", 0))
            relevant_pages.add(chunk.get("page_end", 0))

    # ── Pass 3: Co-location resolution (highest confidence) ──────────
    colocated_image_ids: Set[str] = set()
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            linked = chunk.get("linked_images", [])
            if isinstance(linked, list):
                colocated_image_ids.update(linked)

    # ── Generate query vectors ───────────────────────────────────────
    query_text_vector = await _embed_query_text(query)
    query_clip_vector = _get_clip_text_vector(query)

    # ── Run image search ─────────────────────────────────────────────
    from app.retrieval.image_searcher import ImageSearcher

    searcher = ImageSearcher()

    try:
        image_results = searcher.search(
            query_text_vector=query_text_vector,
            query_clip_vector=query_clip_vector,
            relevant_page_numbers=relevant_pages,
            top_k=settings.IMAGE_RETURN_TOP_K,
            min_score=settings.CLIP_SIMILARITY_THRESHOLD,
        )
    except Exception as e:
        logger.error("image_search_failed", error=str(e))
        image_results = []

    # ── Convert to state dict format ─────────────────────────────────
    retrieved_images: List[Dict[str, Any]] = []

    for img in image_results:
        retrieved_images.append({
            "image_id": img.image_id,
            "doc_id": img.doc_id,
            "page_number": img.page_number,
            "caption": img.caption,
            "description": img.description,
            "image_type": img.image_type,
            "storage_url": img.storage_url,
            "thumbnail_url": img.thumbnail_url,
            "source_file": img.source_file,
            "composite_score": img.composite_score,
            "clip_score": img.clip_score,
            "caption_score": img.caption_score,
        })

    # If co-located images weren't found by the search, mark them
    if colocated_image_ids:
        existing_ids = {img["image_id"] for img in retrieved_images}
        missing_colocated = colocated_image_ids - existing_ids
        if missing_colocated:
            logger.debug(
                "colocated_images_not_in_search",
                count=len(missing_colocated),
            )

    logger.info(
        "visual_specialist_completed",
        images_found=len(retrieved_images),
        colocated_ids=len(colocated_image_ids),
    )

    return {
        "retrieved_images": retrieved_images,
    }


async def _embed_query_text(query: str) -> List[float] | None:
    """Embed query text using shared TextEmbedder singleton for caption similarity."""
    try:
        from app.core.shared import get_text_embedder
        embedder = get_text_embedder()
        return embedder.embed_text(query)
    except Exception as e:
        logger.warning("query_text_embedding_failed", error=str(e))
        return None


def _get_clip_text_vector(query: str) -> List[float] | None:
    """Encode query text with CLIP text encoder for cross-modal search."""
    try:
        from app.ingestion.image_processor import ImageProcessor

        processor = ImageProcessor()
        clip_vec = processor.generate_clip_text_embedding(query)

        # Check if it's a dummy (all zeros)
        if clip_vec and any(v != 0.0 for v in clip_vec):
            return clip_vec

        return None

    except Exception as e:
        logger.warning("clip_text_encoding_failed", error=str(e))
        return None

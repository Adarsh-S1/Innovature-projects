"""
Cross-Modal Linker — establishes links between text chunks and images
using three complementary strategies:

1. Spatial Co-location: Same page or ±1 page proximity
2. Caption Embedding Match: Cosine similarity between chunk and caption vectors
3. Explicit Reference Parsing: Regex for "See Figure 3", "refer to diagram 2B"
"""

import re
from typing import Dict, List, Set, Tuple

import numpy as np

from app.core.logging import get_logger
from app.models.domain import ImageRecord, TextChunk

logger = get_logger(__name__)

# Patterns for explicit figure references in text
_FIGURE_REFERENCE_PATTERNS = [
    re.compile(
        r'(?:see|refer\s+to|shown\s+in|illustrated\s+in|as\s+(?:shown|depicted)\s+in)\s+'
        r'(?:figure|fig\.?|diagram|image|photo|schematic|chart|table)\s*\.?\s*'
        r'(\d+[A-Za-z]?(?:\.\d+)?)',
        re.IGNORECASE,
    ),
    re.compile(
        r'(?:figure|fig\.?|diagram|image|schematic|chart)\s*\.?\s*'
        r'(\d+[A-Za-z]?(?:\.\d+)?)',
        re.IGNORECASE,
    ),
]


class CrossModalLinker:
    """
    Links text chunks to related images using multiple strategies.
    """

    def __init__(self, page_proximity: int = 1):
        """
        Args:
            page_proximity: Number of pages ± to consider for co-location.
                           Default is 1 (same page or adjacent pages).
        """
        self._page_proximity = page_proximity

    def link_chunks_and_images(
        self,
        chunks: List[TextChunk],
        images: List[ImageRecord],
    ) -> Tuple[List[TextChunk], List[ImageRecord]]:
        """
        Establish bidirectional links between text chunks and images.

        Applies all three linking strategies and merges results.

        Args:
            chunks: List of text chunks from the document.
            images: List of image records from the document.

        Returns:
            Tuple of (updated_chunks, updated_images) with linked IDs populated.
        """
        if not chunks or not images:
            return chunks, images

        logger.info(
            "cross_modal_linking_started",
            num_chunks=len(chunks),
            num_images=len(images),
        )

        # Strategy 1: Spatial co-location (highest confidence)
        colocation_links = self._spatial_colocation(chunks, images)

        # Strategy 2: Explicit reference parsing
        reference_links = self._explicit_reference_parsing(chunks, images)

        # Strategy 3: Caption embedding similarity (if vectors are available)
        embedding_links = self._caption_embedding_match(chunks, images)

        # Merge all links (union, deduplicated)
        all_links = self._merge_links(colocation_links, reference_links, embedding_links)

        # Apply links to chunk and image objects
        chunks, images = self._apply_links(chunks, images, all_links)

        total_links = sum(len(c.linked_images) for c in chunks)
        logger.info(
            "cross_modal_linking_completed",
            total_links=total_links,
        )

        return chunks, images

    def _spatial_colocation(
        self, chunks: List[TextChunk], images: List[ImageRecord]
    ) -> Dict[str, Set[str]]:
        """
        Strategy 1: Link chunks and images on the same page or ±N pages.
        Returns dict mapping chunk_id → set of image_ids.
        """
        links: Dict[str, Set[str]] = {}

        # Build page → images lookup
        page_to_images: Dict[int, List[str]] = {}
        for img in images:
            page_to_images.setdefault(img.page_number, []).append(img.image_id)

        for chunk in chunks:
            chunk_links: Set[str] = set()

            # Check pages in the chunk's range ± proximity
            for page_num in range(
                chunk.page_start - self._page_proximity,
                chunk.page_end + self._page_proximity + 1,
            ):
                if page_num in page_to_images:
                    chunk_links.update(page_to_images[page_num])

            if chunk_links:
                links[chunk.chunk_id] = chunk_links

        return links

    def _explicit_reference_parsing(
        self, chunks: List[TextChunk], images: List[ImageRecord]
    ) -> Dict[str, Set[str]]:
        """
        Strategy 3: Parse text for explicit figure references like
        "See Figure 3", "refer to diagram 2B".

        Matches references to images by page proximity.
        """
        links: Dict[str, Set[str]] = {}

        for chunk in chunks:
            # Find all figure references in chunk text
            figure_refs: List[str] = []
            for pattern in _FIGURE_REFERENCE_PATTERNS:
                figure_refs.extend(pattern.findall(chunk.text))

            if not figure_refs:
                continue

            # For each reference, find the best matching image
            # (closest to the chunk's page range)
            chunk_links: Set[str] = set()
            for _ref in figure_refs:
                # Find images on nearby pages
                for img in images:
                    page_dist = min(
                        abs(img.page_number - chunk.page_start),
                        abs(img.page_number - chunk.page_end),
                    )
                    if page_dist <= self._page_proximity + 2:
                        chunk_links.add(img.image_id)

            if chunk_links:
                links[chunk.chunk_id] = chunk_links

        return links

    def _caption_embedding_match(
        self,
        chunks: List[TextChunk],
        images: List[ImageRecord],
        threshold: float = 0.40,
    ) -> Dict[str, Set[str]]:
        """
        Strategy 2: Compute cosine similarity between chunk text vectors
        and image caption vectors. Link pairs above threshold.

        Only works if both text_vector and caption_vector are populated.
        """
        links: Dict[str, Set[str]] = {}

        # Check if vectors are available
        chunks_with_vectors = [c for c in chunks if c.text_vector]
        images_with_vectors = [i for i in images if i.caption_vector]

        if not chunks_with_vectors or not images_with_vectors:
            return links

        # Build caption vector matrix
        caption_matrix = np.array([img.caption_vector for img in images_with_vectors])
        image_ids = [img.image_id for img in images_with_vectors]

        for chunk in chunks_with_vectors:
            chunk_vector = np.array(chunk.text_vector)

            # Cosine similarity
            norms = np.linalg.norm(caption_matrix, axis=1) * np.linalg.norm(chunk_vector)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            similarities = np.dot(caption_matrix, chunk_vector) / norms

            # Find images above threshold
            matched_indices = np.where(similarities >= threshold)[0]
            if len(matched_indices) > 0:
                chunk_links = {image_ids[i] for i in matched_indices}
                links[chunk.chunk_id] = chunk_links

        return links

    def _merge_links(self, *link_dicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Merge multiple link dictionaries (union of image sets per chunk)."""
        merged: Dict[str, Set[str]] = {}

        for link_dict in link_dicts:
            for chunk_id, image_ids in link_dict.items():
                if chunk_id not in merged:
                    merged[chunk_id] = set()
                merged[chunk_id].update(image_ids)

        return merged

    def _apply_links(
        self,
        chunks: List[TextChunk],
        images: List[ImageRecord],
        links: Dict[str, Set[str]],
    ) -> Tuple[List[TextChunk], List[ImageRecord]]:
        """Apply computed links to chunk and image objects (bidirectional)."""
        # Build reverse mapping: image_id → set of chunk_ids
        reverse_links: Dict[str, Set[str]] = {}
        for chunk_id, image_ids in links.items():
            for image_id in image_ids:
                if image_id not in reverse_links:
                    reverse_links[image_id] = set()
                reverse_links[image_id].add(chunk_id)

        # Apply to chunks
        chunk_map = {c.chunk_id: c for c in chunks}
        for chunk_id, image_ids in links.items():
            if chunk_id in chunk_map:
                existing = set(chunk_map[chunk_id].linked_images)
                existing.update(image_ids)
                chunk_map[chunk_id].linked_images = list(existing)

        # Apply to images
        image_map = {i.image_id: i for i in images}
        for image_id, chunk_ids in reverse_links.items():
            if image_id in image_map:
                existing = set(image_map[image_id].linked_chunk_ids)
                existing.update(chunk_ids)
                image_map[image_id].linked_chunk_ids = list(existing)

        return list(chunk_map.values()), list(image_map.values())

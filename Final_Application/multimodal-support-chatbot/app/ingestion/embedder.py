"""
Text Embedder — wrapper around OpenAI's text-embedding-3-large model.

Provides:
- Single text embedding
- Batch embedding with configurable batch size
- Retry logic for API failures
"""

from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class TextEmbedder:
    """
    Generates text embeddings using OpenAI text-embedding-3-large.
    Supports batch processing and automatic retries.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = None

    def _ensure_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._settings.OPENAI_API_KEY,
                max_retries=self._settings.OPENAI_MAX_RETRIES,
                timeout=self._settings.OPENAI_TIMEOUT,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Text to embed.

        Returns:
            List of floats (3072-dimensional vector).

        Raises:
            EmbeddingError: If embedding generation fails after retries.
        """
        self._ensure_client()

        try:
            response = self._client.embeddings.create(
                model=self._settings.OPENAI_EMBEDDING_MODEL,
                input=text,
                dimensions=self._settings.OPENAI_EMBEDDING_DIMENSIONS,
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error("text_embedding_failed", error=str(e), text_length=len(text))
            raise EmbeddingError(f"Failed to generate text embedding: {e}")

    def embed_batch(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Splits into sub-batches of `batch_size` to respect API limits.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per API call. Defaults to settings.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        self._ensure_client()
        batch_size = batch_size or self._settings.BATCH_EMBED_SIZE
        all_embeddings: List[List[float]] = []

        logger.info(
            "batch_embedding_started",
            total_texts=len(texts),
            batch_size=batch_size,
        )

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_sub_batch(batch)
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                "batch_progress",
                completed=min(i + batch_size, len(texts)),
                total=len(texts),
            )

        logger.info(
            "batch_embedding_completed",
            total_embeddings=len(all_embeddings),
        )

        return all_embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _embed_sub_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a sub-batch of texts in a single API call."""
        try:
            # Truncate very long texts to avoid token limits
            cleaned_texts = [t[:8000] if len(t) > 8000 else t for t in texts]

            response = self._client.embeddings.create(
                model=self._settings.OPENAI_EMBEDDING_MODEL,
                input=cleaned_texts,
                dimensions=self._settings.OPENAI_EMBEDDING_DIMENSIONS,
            )

            # Sort by index to ensure correct ordering
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]

        except Exception as e:
            logger.error(
                "sub_batch_embedding_failed",
                batch_size=len(texts),
                error=str(e),
            )
            raise EmbeddingError(f"Failed to embed batch of {len(texts)} texts: {e}")

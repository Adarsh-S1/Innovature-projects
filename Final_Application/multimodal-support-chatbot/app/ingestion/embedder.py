"""
Text Embedder — wrapper around local SentenceTransformers model.

Provides:
- Single text embedding
- Batch embedding with configurable batch size
"""

from typing import List, Optional

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class TextEmbedder:
    """
    Generates text embeddings using a local Hugging Face SentenceTransformers model.
    """

    def __init__(self):
        self._settings = get_settings()
        self._model = None

    def _ensure_model(self):
        """Lazy-load SentenceTransformer model."""
        if self._model is None:
            logger.info("loading_sentence_transformer", model=self._settings.TEXT_EMBEDDING_MODEL)
            try:
                from sentence_transformers import SentenceTransformer
                # Will automatically download on first run
                self._model = SentenceTransformer(self._settings.TEXT_EMBEDDING_MODEL)
            except Exception as e:
                logger.error("model_load_failed", error=str(e))
                raise EmbeddingError(f"Failed to load sentence transformer model: {e}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Text to embed.

        Returns:
            List of floats (384-dimensional vector).

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        self._ensure_model()

        try:
            embedding = self._model.encode(text, normalize_embeddings=True)
            return embedding.tolist()

        except Exception as e:
            logger.error("text_embedding_failed", error=str(e), text_length=len(text))
            raise EmbeddingError(f"Failed to generate text embedding: {e}")

    def embed_batch(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per forward pass.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        self._ensure_model()
        batch_size = batch_size or self._settings.BATCH_EMBED_SIZE

        logger.info(
            "batch_embedding_started",
            total_texts=len(texts),
            batch_size=batch_size,
        )

        try:
            embeddings = self._model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
            logger.info(
                "batch_embedding_completed",
                total_embeddings=len(embeddings),
            )
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            logger.error("batch_embedding_failed", error=str(e))
            raise EmbeddingError(f"Failed to embed batch of {len(texts)} texts: {e}")

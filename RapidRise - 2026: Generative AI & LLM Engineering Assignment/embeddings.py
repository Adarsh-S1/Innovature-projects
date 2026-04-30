"""
embeddings.py — Embedding model wrapper using HuggingFace sentence-transformers.

Uses the all-MiniLM-L6-v2 model to generate 384-dimensional embeddings
for text chunks and queries.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import config

# Global model instance (loaded once)
_model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model: {config.EMBEDDING_MODEL_NAME}...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        print(f"[Embeddings] Model loaded. Dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for a single text string.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding (384 dimensions).
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts: List of input texts to embed.
        batch_size: Number of texts to process at once.

    Returns:
        List of embedding vectors.
    """
    model = _get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "The Transformer uses self-attention mechanisms.",
        "BERT is a bidirectional language model.",
    ]
    vectors = embed_batch(test_texts)
    print(f"Generated {len(vectors)} embeddings of dimension {len(vectors[0])}")

    # Test similarity
    from numpy.linalg import norm
    v1, v2 = np.array(vectors[0]), np.array(vectors[1])
    similarity = np.dot(v1, v2) / (norm(v1) * norm(v2))
    print(f"Cosine similarity: {similarity:.4f}")

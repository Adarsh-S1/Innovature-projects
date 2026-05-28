"""
Shared service utilities to eliminate code duplication across agents and modules.

Provides:
- LLM client singleton (Groq AsyncGroq / Groq)
- TextEmbedder singleton
- JSON field parsing
- Markdown fence stripping
"""

import json
import re
from functools import lru_cache
from typing import Any, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Groq LLM Client Singleton ───────────────────────────────────────────

_async_groq_client = None
_sync_groq_client = None


def get_async_groq_client():
    """
    Get or create a shared AsyncGroq client instance.

    Uses lazy initialization with module-level caching to avoid creating
    multiple client instances across agents (context_router, answer_synthesizer,
    quality_guard).
    """
    global _async_groq_client

    if _async_groq_client is None:
        from groq import AsyncGroq
        settings = get_settings()

        _async_groq_client = AsyncGroq(
            api_key=settings.GROQ_API_KEY,
            max_retries=2,
            timeout=30,
        )
        logger.debug("async_groq_client_initialized")

    return _async_groq_client


def get_sync_groq_client():
    """
    Get or create a shared synchronous Groq client instance.

    Used by components running in synchronous contexts (e.g. Celery tasks,
    image_processor).
    """
    global _sync_groq_client

    if _sync_groq_client is None:
        from groq import Groq
        settings = get_settings()

        _sync_groq_client = Groq(
            api_key=settings.GROQ_API_KEY,
            max_retries=1,
            timeout=30,
        )
        logger.debug("sync_groq_client_initialized")

    return _sync_groq_client


# ── TextEmbedder Singleton ──────────────────────────────────────────────

_text_embedder = None


def get_text_embedder():
    """
    Get or create a shared TextEmbedder instance.

    The SentenceTransformer model is expensive to load from disk (~200ms).
    This singleton ensures the model is loaded only once and reused across
    hybrid_search, visual_specialist, and ingestion agents.
    """
    global _text_embedder

    if _text_embedder is None:
        from app.ingestion.embedder import TextEmbedder
        _text_embedder = TextEmbedder()
        logger.debug("text_embedder_singleton_initialized")

    return _text_embedder


# ── JSON Field Parsing ──────────────────────────────────────────────────


def parse_json_field(value: Any, default: Optional[Any] = None) -> Any:
    """
    Parse a JSON-serialized Milvus VARCHAR field into its Python equivalent.

    Milvus stores lists/dicts as JSON strings in VARCHAR fields.
    This safely parses them, with a fallback default.

    Args:
        value: The raw value from Milvus (could be str, list, dict, or None).
        default: Default value if parsing fails (defaults to empty list).

    Returns:
        Parsed Python object, or the default value on failure.
    """
    if default is None:
        default = []

    if value is None:
        return default

    # Already parsed (e.g. in offline/test mode)
    if isinstance(value, (list, dict)):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            logger.debug(
                "json_field_parse_failed",
                value_preview=value[:50] if value else "",
            )
            return default

    return default


# ── Markdown Fence Stripping ────────────────────────────────────────────


def strip_markdown_fences(content: str) -> str:
    """
    Strip markdown code fences (```json ... ```) from LLM responses.

    Many LLMs wrap JSON responses in markdown fences even when asked
    not to. This utility safely removes them.

    Args:
        content: Raw LLM response string.

    Returns:
        Clean content string with fences removed.
    """
    content = content.strip()

    if content.startswith("```"):
        # Remove opening fence (may include language tag like ```json)
        content = content.split("\n", 1)[-1]
        # Remove closing fence
        content = content.rsplit("```", 1)[0]

    return content.strip()

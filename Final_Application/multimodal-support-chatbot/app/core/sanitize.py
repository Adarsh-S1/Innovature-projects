"""
Input sanitization utilities to prevent injection attacks.

Provides sanitization for:
- Milvus filter expressions (prevents expression injection)
- General string inputs
"""

import re
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# Characters that could be used to break out of a Milvus string literal
# or inject operators into filter expressions
_MILVUS_UNSAFE_CHARS = re.compile(r'["\\\x00|&!=<>]')

# Valid identifier pattern (doc_ids, session_ids, etc.)
_VALID_IDENTIFIER = re.compile(r'^[a-zA-Z0-9_\-.:]+$')


def sanitize_milvus_value(value: str) -> str:
    """
    Sanitize a string value for safe use in Milvus filter expressions.

    Strips characters that could break out of a quoted string literal
    and validates the input is a reasonable identifier.

    Args:
        value: Raw user-provided string to use in a Milvus expression.

    Returns:
        Sanitized string safe for use in Milvus expressions.

    Raises:
        ValueError: If the input contains malicious patterns or is too long.
    """
    if not value or not value.strip():
        raise ValueError("Empty value is not allowed in Milvus expressions")

    value = value.strip()

    # Length limit to prevent oversized expressions
    if len(value) > 512:
        raise ValueError(
            f"Value too long for Milvus expression ({len(value)} chars, max 512)"
        )

    # Remove or escape dangerous characters
    sanitized = _MILVUS_UNSAFE_CHARS.sub("", value)

    if sanitized != value:
        logger.warning(
            "milvus_value_sanitized",
            original_length=len(value),
            sanitized_length=len(sanitized),
        )

    return sanitized


def validate_identifier(value: str, field_name: str = "identifier") -> str:
    """
    Validate that a value is a safe identifier (alphanumeric, hyphens,
    underscores, dots, colons).

    Used for doc_ids, session_ids, product_ids, etc.

    Args:
        value: The identifier to validate.
        field_name: Name of the field (for error messages).

    Returns:
        The validated identifier string.

    Raises:
        ValueError: If the identifier contains invalid characters.
    """
    if not value or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")

    value = value.strip()

    if len(value) > 256:
        raise ValueError(
            f"{field_name} too long ({len(value)} chars, max 256)"
        )

    if not _VALID_IDENTIFIER.match(value):
        raise ValueError(
            f"{field_name} contains invalid characters. "
            "Only alphanumeric, hyphens, underscores, dots, and colons are allowed."
        )

    return value


def build_milvus_filter(
    doc_id: Optional[str] = None,
    language: Optional[str] = None,
) -> Optional[str]:
    """
    Safely build a Milvus filter expression from optional parameters.

    Centralizes filter construction to prevent expression injection.

    Args:
        doc_id: Optional document ID filter.
        language: Optional language filter.

    Returns:
        A safe Milvus filter expression string, or None if no filters.
    """
    parts = []

    if doc_id:
        safe_doc_id = sanitize_milvus_value(doc_id)
        parts.append(f'doc_id == "{safe_doc_id}"')

    if language:
        safe_lang = sanitize_milvus_value(language)
        parts.append(f'language == "{safe_lang}"')

    if not parts:
        return None

    return " && ".join(parts)

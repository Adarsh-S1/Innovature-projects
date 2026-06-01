"""
Custom exception classes for the multimodal support chatbot.
These provide structured error handling across all application layers.
"""

from typing import Any, Dict, Optional


class ChatbotBaseException(Exception):
    """Base exception for all chatbot-specific errors."""

    def __init__(
        self,
        message: str,
        detail: Optional[str] = None,
        status_code: int = 500,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.detail = detail or message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exception for API error responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "detail": self.detail,
            "context": self.context,
        }


# ── Ingestion Errors ────────────────────────────────────────────────────


class IngestionError(ChatbotBaseException):
    """Raised when PDF ingestion or processing fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=422, **kwargs)


class PDFParsingError(IngestionError):
    """Raised when PDF text/image extraction fails."""
    pass


class ChunkingError(IngestionError):
    """Raised when text chunking encounters an error."""
    pass


class ImageProcessingError(IngestionError):
    """Raised when image extraction or CLIP embedding fails."""
    pass


class EmbeddingError(IngestionError):
    """Raised when text or image embedding generation fails."""
    pass


# ── Retrieval Errors ────────────────────────────────────────────────────


class RetrievalError(ChatbotBaseException):
    """Raised when search or retrieval operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=502, **kwargs)


class VectorSearchError(RetrievalError):
    """Raised when Milvus vector search fails."""
    pass


class RerankingError(RetrievalError):
    """Raised when cross-encoder reranking fails."""
    pass


class ImageRetrievalError(RetrievalError):
    """Raised when CLIP-based image retrieval fails."""
    pass


# ── Agent Errors ────────────────────────────────────────────────────────


class AgentError(ChatbotBaseException):
    """Raised when an agent in the LangGraph pipeline fails."""

    def __init__(self, message: str, agent_name: Optional[str] = None, **kwargs):
        self.agent_name = agent_name
        context = kwargs.pop("context", {})
        context["agent_name"] = agent_name
        super().__init__(message, status_code=500, context=context, **kwargs)


class ContextRouterError(AgentError):
    """Raised when intent classification / routing fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, agent_name="context_router", **kwargs)


class QualityGuardError(AgentError):
    """Raised when quality validation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, agent_name="quality_guard", **kwargs)


class AnswerSynthesisError(AgentError):
    """Raised when answer generation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, agent_name="answer_synthesizer", **kwargs)


# ── Database / Infrastructure Errors ────────────────────────────────────


class DatabaseError(ChatbotBaseException):
    """Raised when a database operation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=503, **kwargs)


class MilvusConnectionError(DatabaseError):
    """Raised when connection to Milvus fails."""
    pass


class RedisConnectionError(DatabaseError):
    """Raised when connection to Redis fails."""
    pass


class RustFSError(DatabaseError):
    """Raised when RustFS object storage operations fail."""
    pass


# ── API / Session Errors ────────────────────────────────────────────────


class AuthenticationError(ChatbotBaseException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(ChatbotBaseException):
    """Raised when authorization fails."""

    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class RateLimitError(ChatbotBaseException):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, status_code=429, **kwargs)


class SessionNotFoundError(ChatbotBaseException):
    """Raised when a session ID does not exist."""

    def __init__(self, session_id: str, **kwargs):
        super().__init__(
            f"Session '{session_id}' not found",
            status_code=404,
            context={"session_id": session_id},
            **kwargs,
        )


class DocumentNotFoundError(ChatbotBaseException):
    """Raised when a document ID does not exist."""

    def __init__(self, doc_id: str, **kwargs):
        super().__init__(
            f"Document '{doc_id}' not found",
            status_code=404,
            context={"doc_id": doc_id},
            **kwargs,
        )


# ── LLM Errors ──────────────────────────────────────────────────────────


class LLMError(ChatbotBaseException):
    """Raised when LLM API call fails after retries."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=502, **kwargs)


class LLMRateLimitError(LLMError):
    """Raised when LLM provider rate-limits us."""

    def __init__(self, message: str = "LLM rate limit exceeded", **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = 429


class LLMTimeoutError(LLMError):
    """Raised when LLM call times out."""

    def __init__(self, message: str = "LLM request timed out", **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = 504

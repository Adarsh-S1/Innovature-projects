"""
Application configuration using pydantic-settings.
Loads settings from environment variables and .env file.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the multimodal support chatbot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    APP_NAME: str = "Multimodal Support Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = Field(default="development", description="development | staging | production")
    LOG_LEVEL: str = Field(default="INFO", description="DEBUG | INFO | WARNING | ERROR | CRITICAL")

    # ── API Server ───────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000", "http://localhost:8000"])
    RATE_LIMIT_PER_MINUTE: int = 60

    # ── Authentication ───────────────────────────────────────────────────
    API_KEY: Optional[str] = Field(default=None, description="API key for simple auth")
    JWT_SECRET_KEY: Optional[str] = Field(default=None, description="JWT signing secret")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60

    # ── Text Models (Groq + Local Embeddings) ────────────────────────────
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key (deprecated)")
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_DIMENSIONS: int = 384
    OPENAI_MAX_RETRIES: int = 3
    OPENAI_TIMEOUT: int = 60

    # ── Google Gemini (Fallback LLM) ─────────────────────────────────────
    GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Google AI API key for Gemini fallback")
    GEMINI_MODEL: str = "gemini-1.5-pro"

    # ── CLIP ─────────────────────────────────────────────────────────────
    CLIP_MODEL_NAME: str = "ViT-L/14"
    CLIP_DEVICE: str = "cuda"
    CLIP_EMBEDDING_DIMENSIONS: int = 768

    # ── Milvus ───────────────────────────────────────────────────────────
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    MILVUS_DATABASE: str = "default"
    MILVUS_TEXT_COLLECTION: str = "text_collection"
    MILVUS_IMAGE_COLLECTION: str = "image_collection"

    # ── Redis ────────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_SESSION_TTL_SECONDS: int = 3600  # 1 hour
    REDIS_CACHE_TTL_SECONDS: int = 300     # 5 minutes

    # ── MinIO (S3-compatible) ────────────────────────────────────────────
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "multimodal-chatbot"
    MINIO_SECURE: bool = False
    MINIO_PRESIGNED_URL_TTL: int = 3600  # 1 hour

    # ── Ingestion Pipeline ───────────────────────────────────────────────
    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 128
    MIN_IMAGE_WIDTH: int = 100
    MIN_IMAGE_HEIGHT: int = 100
    MAX_CAPTION_LENGTH: int = 200
    BATCH_EMBED_SIZE: int = 32

    # ── Retrieval ────────────────────────────────────────────────────────
    TEXT_SEARCH_TOP_K: int = 20
    TEXT_RERANK_TOP_K: int = 5
    IMAGE_SEARCH_TOP_K: int = 10
    IMAGE_RETURN_TOP_K: int = 3
    CLIP_SIMILARITY_THRESHOLD: float = 0.25
    CAPTION_SIMILARITY_THRESHOLD: float = 0.70

    # ── Quality Guard ────────────────────────────────────────────────────
    QUALITY_THRESHOLD: float = 0.75
    MAX_RETRIES: int = 2

    # ── Celery ───────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── Observability ────────────────────────────────────────────────────
    OTEL_EXPORTER_ENDPOINT: Optional[str] = None
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "multimodal-chatbot"

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v_upper

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v.lower()

    @property
    def redis_url(self) -> str:
        """Construct full Redis URL."""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def milvus_uri(self) -> str:
        """Construct Milvus connection URI."""
        return f"http://{self.MILVUS_HOST}:{self.MILVUS_PORT}"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — call this everywhere."""
    return Settings()

"""
Pydantic response schemas for the API layer.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ImageResult(BaseModel):
    """A single image returned as part of a chat response."""

    image_id: str = Field(description="Unique identifier of the image.")
    url: str = Field(description="Presigned URL to the full-resolution image.")
    thumbnail_url: Optional[str] = Field(default=None, description="Presigned URL to the thumbnail.")
    caption: str = Field(description="AI-generated short caption.")
    relevance_score: float = Field(description="Composite relevance score (0-1).")
    image_type: str = Field(description="Image classification type.")
    source: str = Field(description="Source reference.")


class SourceReference(BaseModel):
    """Citation for a retrieved text chunk."""

    doc_id: str
    source_file: str
    section_path: List[str] = Field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    relevance_score: float = 0.0


class ChatResponse(BaseModel):
    """Full response from the chat endpoint."""

    session_id: str
    answer: str
    images: List[ImageResult] = Field(default_factory=list)
    sources: List[SourceReference] = Field(default_factory=list)
    confidence_level: str = "high"
    quality_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Response after initiating PDF ingestion."""

    job_id: str
    status: str = "queued"
    message: str = ""


class IngestStatusResponse(BaseModel):
    """Status of a PDF ingestion job."""

    job_id: str
    status: str = "queued"
    progress: float = 0.0
    total_chunks: int = 0
    total_images: int = 0
    errors: List[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Information about an indexed document."""

    doc_id: str
    source_file: str
    product_id: Optional[str] = None
    language: str = "en"
    total_chunks: int = 0
    total_images: int = 0
    ingested_at: str = ""
    tags: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = ""
    services: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    message: str
    detail: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

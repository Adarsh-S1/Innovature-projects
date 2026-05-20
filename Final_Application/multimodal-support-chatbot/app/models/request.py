"""
Pydantic request schemas for the API layer.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the main chat endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's natural language query.",
        examples=["Where is the RAM slot on board X?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID. Auto-created if omitted.",
    )
    product_filter: Optional[str] = Field(
        default=None,
        description="Filter retrieval to a specific product or document ID.",
    )
    stream: bool = Field(
        default=False,
        description="If true, response is streamed via Server-Sent Events.",
    )


class PDFIngestRequest(BaseModel):
    """Request body (metadata) accompanying a PDF upload for ingestion."""

    product_id: Optional[str] = Field(
        default=None,
        description="Product identifier to tag all chunks from this PDF.",
        examples=["SKU-X100"],
    )
    language: str = Field(
        default="en",
        description="Language of the document content.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags for document categorisation.",
    )
    priority: str = Field(
        default="normal",
        description="Ingestion priority: low | normal | high",
    )

"""
Internal domain models for the multimodal support chatbot.
These are NOT exposed via the API — they represent internal data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    LIST = "list"
    HEADING = "heading"


class ImageType(str, Enum):
    SCHEMATIC = "schematic"
    WIRING_DIAGRAM = "wiring_diagram"
    CHART = "chart"
    PHOTO = "photo"
    TABLE = "table"
    FLOWCHART = "flowchart"
    OTHER = "other"


class QueryIntent(str, Enum):
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    MULTIMODAL = "multimodal"


class QueryType(str, Enum):
    HOW_TO = "how_to"
    TROUBLESHOOT = "troubleshoot"
    LOCATE_COMPONENT = "locate_component"
    SPECIFICATION = "specification"
    GENERAL = "general"


class TextChunk(BaseModel):
    """A single text chunk extracted and embedded from a PDF."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    page_start: int
    page_end: int
    section_path: List[str] = Field(default_factory=list)
    text: str
    text_vector: Optional[List[float]] = None
    linked_images: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_type: ChunkType = ChunkType.PARAGRAPH


class ImageRecord(BaseModel):
    """A single image extracted from a PDF with embeddings and metadata."""

    image_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    page_number: int
    caption: str = ""
    description: str = ""
    topic_concept: str = ""
    keyword_tags: List[str] = Field(default_factory=list)
    image_type: ImageType = ImageType.OTHER
    clip_vector: Optional[List[float]] = None
    caption_vector: Optional[List[float]] = None
    storage_url: str = ""
    thumbnail_url: str = ""
    linked_chunk_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    """Metadata record for an ingested PDF document."""

    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    source_file: str
    product_id: Optional[str] = None
    language: str = "en"
    total_chunks: int = 0
    total_images: int = 0
    tags: List[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "active"


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    images: List[str] = Field(default_factory=list)


class Session(BaseModel):
    """Chat session with conversation history."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    history: List[ConversationTurn] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RoutingResult(BaseModel):
    """Output of Agent 1 — Context Router."""

    intent: QueryIntent = QueryIntent.MULTIMODAL
    needs_image: bool = True
    detected_components: List[str] = Field(default_factory=list)
    query_type: QueryType = QueryType.GENERAL
    extracted_error_codes: List[str] = Field(default_factory=list)
    is_followup: bool = False
    reformulated_query: str = ""
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

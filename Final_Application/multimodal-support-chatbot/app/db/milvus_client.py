"""
Milvus client — manages connection, collections, and CRUD operations.
"""

from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from app.core.config import get_settings
from app.core.exceptions import MilvusConnectionError
from app.core.logging import get_logger

logger = get_logger(__name__)


class MilvusManager:
    """
    Manages Milvus vector database connections and collections.

    Provides methods to:
    - Connect/disconnect from Milvus
    - Create text and image collections with proper schemas
    - Insert, search, and delete vectors
    """

    def __init__(self):
        self._settings = get_settings()
        self._client: Optional[MilvusClient] = None
        self._text_collection: Optional[Collection] = None
        self._image_collection: Optional[Collection] = None

    async def connect(self) -> None:
        """Establish connection to Milvus."""
        try:
            connections.connect(
                alias="default",
                host=self._settings.MILVUS_HOST,
                port=self._settings.MILVUS_PORT,
                user=self._settings.MILVUS_USER or "",
                password=self._settings.MILVUS_PASSWORD or "",
                db_name=self._settings.MILVUS_DATABASE,
            )
            logger.info(
                "milvus_connected",
                host=self._settings.MILVUS_HOST,
                port=self._settings.MILVUS_PORT,
            )
        except Exception as e:
            logger.error("milvus_connection_failed", error=str(e))
            raise MilvusConnectionError(f"Failed to connect to Milvus: {e}")

    async def disconnect(self) -> None:
        """Close Milvus connection."""
        try:
            connections.disconnect("default")
            logger.info("milvus_disconnected")
        except Exception as e:
            logger.warning("milvus_disconnect_error", error=str(e))

    def _build_text_collection_schema(self) -> CollectionSchema:
        """Define schema for the text chunk collection."""
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="page_start", dtype=DataType.INT32),
            FieldSchema(name="page_end", dtype=DataType.INT32),
            FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=8),
            FieldSchema(
                name="text_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._settings.TEXT_EMBEDDING_DIMENSIONS,
            ),
            FieldSchema(
                name="linked_images",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=256,
                max_length=64,
            ),
        ]
        return CollectionSchema(fields=fields, description="Text chunks from PDF manuals")

    def _build_image_collection_schema(self) -> CollectionSchema:
        """Define schema for the image collection."""
        fields = [
            FieldSchema(name="image_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="page_number", dtype=DataType.INT32),
            FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="topic_concept", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="keyword_tags", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="image_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="storage_url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="thumbnail_url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(
                name="linked_chunk_ids",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=256,
                max_length=64,
            ),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(
                name="clip_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._settings.CLIP_EMBEDDING_DIMENSIONS,
            ),
            FieldSchema(
                name="caption_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._settings.TEXT_EMBEDDING_DIMENSIONS,
            ),
        ]
        return CollectionSchema(fields=fields, description="Images extracted from PDF manuals")

    async def ensure_collections(self) -> None:
        """Create text and image collections if they don't exist."""
        text_name = self._settings.MILVUS_TEXT_COLLECTION
        image_name = self._settings.MILVUS_IMAGE_COLLECTION

        # Text collection
        if not utility.has_collection(text_name):
            schema = self._build_text_collection_schema()
            self._text_collection = Collection(name=text_name, schema=schema)
            # HNSW index on text_vector
            self._text_collection.create_index(
                field_name="text_vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            logger.info("milvus_collection_created", collection=text_name)
        else:
            self._text_collection = Collection(name=text_name)
            logger.info("milvus_collection_exists", collection=text_name)

        # Load collection into memory to enable search and delete operations
        self._text_collection.load()

        # Image collection
        if not utility.has_collection(image_name):
            schema = self._build_image_collection_schema()
            self._image_collection = Collection(name=image_name, schema=schema)
            # HNSW index on clip_vector
            self._image_collection.create_index(
                field_name="clip_vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            # HNSW index on caption_vector
            self._image_collection.create_index(
                field_name="caption_vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            logger.info("milvus_collection_created", collection=image_name)
        else:
            self._image_collection = Collection(name=image_name)
            logger.info("milvus_collection_exists", collection=image_name)

        # Load collection into memory to enable search and delete operations
        self._image_collection.load()

    async def delete_document(self, doc_id: str) -> None:
        """Delete all text chunks and image records associated with a doc_id."""
        if self._text_collection:
            try:
                expr = f'doc_id == "{doc_id}"'
                self._text_collection.delete(expr)
                logger.info("deleted_old_text_chunks", doc_id=doc_id)
            except Exception as e:
                logger.warning("failed_to_delete_text_chunks", doc_id=doc_id, error=str(e))
                
        if self._image_collection:
            try:
                expr = f'doc_id == "{doc_id}"'
                self._image_collection.delete(expr)
                logger.info("deleted_old_image_records", doc_id=doc_id)
            except Exception as e:
                logger.warning("failed_to_delete_image_records", doc_id=doc_id, error=str(e))

    async def health_check(self) -> bool:
        """Check if Milvus is reachable."""
        try:
            connections.connect(
                alias="health_check",
                host=self._settings.MILVUS_HOST,
                port=self._settings.MILVUS_PORT,
            )
            connections.disconnect("health_check")
            return True
        except Exception:
            return False

    @property
    def text_collection(self) -> Optional[Collection]:
        return self._text_collection

    @property
    def image_collection(self) -> Optional[Collection]:
        return self._image_collection


# Singleton instance
milvus_manager = MilvusManager()

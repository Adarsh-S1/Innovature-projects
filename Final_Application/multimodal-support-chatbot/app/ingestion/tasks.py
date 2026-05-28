"""
Celery async ingestion tasks — orchestrates the full PDF ingestion pipeline.

Pipeline stages:
1. PDF parsing (text + image extraction)
2. Text chunking (semantic, structure-aware)
3. Image processing (filtering, captioning, CLIP embedding)
4. Text embedding (text-embedding-3-large)
5. Cross-modal linking
6. Storage (Milvus vectors + MinIO objects)
"""

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from celery import Celery

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()

# ── Celery App ───────────────────────────────────────────────────────────

celery_app = Celery(
    "multimodal_chatbot",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


# ── Synchronous Ingestion Pipeline ───────────────────────────────────────
# This can be called directly (without Celery) for testing
# or invoked as a Celery task for async processing.


class IngestionPipeline:
    """
    Orchestrates the full PDF ingestion pipeline.

    Can be run synchronously or dispatched as a Celery task.
    """

    def __init__(self):
        self._settings = get_settings()

    def run(
        self,
        pdf_bytes: bytes,
        filename: str,
        doc_id: Optional[str] = None,
        product_id: Optional[str] = None,
        language: str = "en",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full ingestion pipeline.

        Args:
            pdf_bytes: Raw PDF file bytes.
            filename: Original filename.
            doc_id: Document ID (auto-generated if not provided).
            product_id: Optional product ID to tag all artifacts.
            language: Document language.
            tags: Optional tags for categorisation.

        Returns:
            Dict with ingestion results (doc_id, chunk_count, image_count, etc.)
        """
        doc_id = doc_id or str(uuid4())
        tags = tags or []
        start_time = time.time()

        logger.info(
            "ingestion_pipeline_started",
            doc_id=doc_id,
            filename=filename,
            product_id=product_id,
        )

        result = {
            "doc_id": doc_id,
            "source_file": filename,
            "status": "processing",
            "total_chunks": 0,
            "total_images": 0,
            "errors": [],
        }

        try:
            # ── Stage 1: PDF Parsing ─────────────────────────────────────
            from app.ingestion.pdf_parser import PDFParser

            parser = PDFParser()
            parsed_doc = parser.parse_from_bytes(pdf_bytes, filename)
            logger.info(
                "stage_1_completed",
                doc_id=doc_id,
                pages=parsed_doc.total_pages,
                images=sum(len(p.images) for p in parsed_doc.pages),
            )

            # ── Stage 2: Text Chunking ───────────────────────────────────
            from app.ingestion.chunker import SemanticChunker

            chunker = SemanticChunker()
            chunks = chunker.chunk_document(
                document=parsed_doc,
                doc_id=doc_id,
                product_id=product_id,
                language=language,
            )
            logger.info("stage_2_completed", doc_id=doc_id, chunks=len(chunks))

            # ── Stage 3: Image Processing ────────────────────────────────
            from app.ingestion.image_processor import ImageProcessor

            image_processor = ImageProcessor()
            all_extracted_images = []
            for page in parsed_doc.pages:
                all_extracted_images.extend(page.images)

            image_records = image_processor.process_images(
                extracted_images=all_extracted_images,
                doc_id=doc_id,
                source_file=filename,
                product_id=product_id,
            )
            logger.info(
                "stage_3_completed", doc_id=doc_id, images=len(image_records)
            )

            # ── Stage 4: Text Embedding ──────────────────────────────────
            from app.ingestion.embedder import TextEmbedder

            embedder = TextEmbedder()

            # Embed text chunks
            chunk_texts = [chunk.text for chunk in chunks]
            if chunk_texts:
                try:
                    chunk_embeddings = embedder.embed_batch(chunk_texts)
                    for chunk, embedding in zip(chunks, chunk_embeddings):
                        chunk.text_vector = embedding
                    logger.info("stage_4a_completed", doc_id=doc_id, embedded_chunks=len(chunks))
                except Exception as e:
                    logger.error("text_embedding_failed", doc_id=doc_id, error=str(e))
                    result["errors"].append(f"Text embedding failed: {e}")

            # Embed image captions (for caption_vector field)
            caption_texts = [
                f"{img.caption} {img.description}" for img in image_records
            ]
            if caption_texts:
                try:
                    caption_embeddings = embedder.embed_batch(caption_texts)
                    for img_record, embedding in zip(image_records, caption_embeddings):
                        img_record.caption_vector = embedding
                    logger.info(
                        "stage_4b_completed",
                        doc_id=doc_id,
                        embedded_captions=len(image_records),
                    )
                except Exception as e:
                    logger.error("caption_embedding_failed", doc_id=doc_id, error=str(e))
                    result["errors"].append(f"Caption embedding failed: {e}")

            # ── Stage 5: Cross-Modal Linking ─────────────────────────────
            from app.retrieval.cross_modal_linker import CrossModalLinker

            linker = CrossModalLinker()
            chunks, image_records = linker.link_chunks_and_images(chunks, image_records)
            logger.info("stage_5_completed", doc_id=doc_id)

            # ── Stage 6: Storage ─────────────────────────────────────────
            storage_errors = self._store_results(
                doc_id=doc_id,
                chunks=chunks,
                image_records=image_records,
                filename=filename,
                product_id=product_id,
                language=language,
                tags=tags,
            )
            result["errors"].extend(storage_errors)
            logger.info("stage_6_completed", doc_id=doc_id)

            # ── Finalize ─────────────────────────────────────────────────
            elapsed = round(time.time() - start_time, 2)
            result.update({
                "status": "completed" if not result["errors"] else "completed_with_errors",
                "total_chunks": len(chunks),
                "total_images": len(image_records),
                "total_pages": parsed_doc.total_pages,
                "elapsed_seconds": elapsed,
            })

            logger.info(
                "ingestion_pipeline_completed",
                doc_id=doc_id,
                chunks=len(chunks),
                images=len(image_records),
                elapsed_s=elapsed,
                errors=len(result["errors"]),
            )

        except Exception as e:
            logger.error(
                "ingestion_pipeline_failed",
                doc_id=doc_id,
                error=str(e),
                exc_info=True,
            )
            result["status"] = "failed"
            result["errors"].append(str(e))

        return result

    def _store_results(
        self,
        doc_id: str,
        chunks: list,
        image_records: list,
        filename: str,
        product_id: Optional[str],
        language: str,
        tags: List[str],
    ) -> List[str]:
        """
        Store chunks in Milvus and images in MinIO + Milvus.
        Returns list of error messages (empty if all succeeded).
        """
        errors: List[str] = []

        # Store images in MinIO (raw + thumbnail)
        for img_record in image_records:
            try:
                raw_bytes = img_record.metadata.get("raw_image_bytes", b"")
                thumb_bytes = img_record.metadata.get("thumbnail_bytes", b"")
                ext = img_record.metadata.get("extension", "png")

                if raw_bytes:
                    image_key = f"images/{doc_id}/{img_record.image_id}.{ext}"
                    img_record.storage_url = image_key
                    self._upload_to_minio(raw_bytes, image_key, f"image/{ext}")

                if thumb_bytes:
                    thumb_key = f"thumbs/{doc_id}/{img_record.image_id}_thumb.png"
                    img_record.thumbnail_url = thumb_key
                    self._upload_to_minio(thumb_bytes, thumb_key, "image/png")

                # Remove raw bytes from metadata before Milvus storage
                img_record.metadata.pop("raw_image_bytes", None)
                img_record.metadata.pop("thumbnail_bytes", None)

            except Exception as e:
                errors.append(f"MinIO upload failed for {img_record.image_id}: {e}")

        # Store text chunks in Milvus
        if chunks:
            try:
                self._insert_chunks_to_milvus(chunks)
            except Exception as e:
                errors.append(f"Milvus text insertion failed: {e}")

        # Store image records in Milvus
        if image_records:
            try:
                self._insert_images_to_milvus(image_records)
            except Exception as e:
                errors.append(f"Milvus image insertion failed: {e}")

        # Store document metadata
        try:
            self._store_document_metadata(
                doc_id, filename, product_id, language, tags,
                len(chunks), len(image_records),
            )
        except Exception as e:
            errors.append(f"Document metadata storage failed: {e}")

        return errors

    def _insert_chunks_to_milvus(self, chunks: list) -> None:
        """Prepare and insert text chunks into Milvus text_collection."""
        from app.db.milvus_client import milvus_manager

        collection = milvus_manager.text_collection
        if collection is None:
            logger.warning("milvus_text_collection_not_available")
            return

        data = []
        for chunk in chunks:
            if chunk.text_vector is None:
                continue

            data.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text[:65535],  # Respect VARCHAR limit
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": json.dumps(chunk.section_path),
                "chunk_type": chunk.chunk_type.value,
                "source_file": chunk.metadata.get("source_file", ""),
                "product_id": chunk.metadata.get("product_id", ""),
                "language": chunk.metadata.get("language", "en"),
                "text_vector": chunk.text_vector,
                "linked_images": json.dumps(chunk.linked_images),
            })

        if data:
            collection.insert(data)
            collection.flush()
            logger.info("milvus_chunks_inserted", count=len(data))

    def _insert_images_to_milvus(self, image_records: list) -> None:
        """Prepare and insert image records into Milvus image_collection."""
        from app.db.milvus_client import milvus_manager

        collection = milvus_manager.image_collection
        if collection is None:
            logger.warning("milvus_image_collection_not_available")
            return

        data = []
        for img in image_records:
            if img.clip_vector is None or img.caption_vector is None:
                continue

            data.append({
                "image_id": img.image_id,
                "doc_id": img.doc_id,
                "page_number": img.page_number,
                "caption": img.caption[:512],
                "description": img.description[:2048],
                "topic_concept": img.topic_concept[:256],
                "keyword_tags": json.dumps(img.keyword_tags)[:1024],
                "image_type": img.image_type.value,
                "storage_url": img.storage_url[:512],
                "thumbnail_url": img.thumbnail_url[:512],
                "linked_chunk_ids": json.dumps(img.linked_chunk_ids)[:2048],
                "source_file": img.metadata.get("source_file", ""),
                "product_id": img.metadata.get("product_id", ""),
                "clip_vector": img.clip_vector,
                "caption_vector": img.caption_vector,
            })

        if data:
            collection.insert(data)
            collection.flush()
            logger.info("milvus_images_inserted", count=len(data))

    def _upload_to_minio(
        self,
        file_data: bytes,
        object_key: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload a file to MinIO, bridging async for Celery compatibility."""
        import asyncio
        from app.db.minio_client import minio_manager

        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(
                minio_manager.upload_file(file_data, object_key, content_type)
            )
            loop.close()
        except Exception as e:
            logger.warning(
                "minio_upload_failed",
                object_key=object_key,
                error=str(e),
            )
            raise

    def _store_document_metadata(
        self,
        doc_id: str,
        filename: str,
        product_id: Optional[str],
        language: str,
        tags: List[str],
        total_chunks: int,
        total_images: int,
    ) -> None:
        """Store document-level metadata in Redis for quick lookup."""
        import asyncio
        from datetime import datetime
        from app.db.redis_client import redis_manager

        metadata = {
            "doc_id": doc_id,
            "source_file": filename,
            "product_id": product_id or "",
            "language": language,
            "tags": tags,
            "total_chunks": total_chunks,
            "total_images": total_images,
            "ingested_at": datetime.utcnow().isoformat(),
            "status": "active",
        }

        try:
            # Use sync approach for Celery compatibility
            loop = asyncio.new_event_loop()
            loop.run_until_complete(
                redis_manager.save_document(doc_id, metadata)
            )
            loop.close()
        except Exception as e:
            logger.warning("document_metadata_storage_failed", error=str(e))


# ── Celery Task ──────────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="ingest_pdf_task",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def ingest_pdf_task(
    self,
    pdf_path: str,
    filename: str,
    doc_id: Optional[str] = None,
    product_id: Optional[str] = None,
    language: str = "en",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Celery task: ingest a PDF file from disk path.

    The PDF is expected to be saved to a temp location by the API endpoint
    before dispatching this task.
    """
    logger.info(
        "celery_ingest_task_started",
        task_id=self.request.id,
        filename=filename,
    )

    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
    except Exception as e:
        logger.error("celery_pdf_read_failed", path=pdf_path, error=str(e))
        return {"status": "failed", "error": str(e)}

    pipeline = IngestionPipeline()
    result = pipeline.run(
        pdf_bytes=pdf_bytes,
        filename=filename,
        doc_id=doc_id,
        product_id=product_id,
        language=language,
        tags=tags,
    )

    # Clean up temp file
    try:
        os.remove(pdf_path)
    except OSError:
        pass

    return result


# ── Convenience: Singleton Pipeline Instance ─────────────────────────────

ingestion_pipeline = IngestionPipeline()

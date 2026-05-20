"""
Document management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional

from app.api.dependencies import verify_api_key
from app.core.logging import get_logger
from app.models.response import DocumentInfo
from app.db.redis_client import redis_manager
from app.db.milvus_client import milvus_manager

logger = get_logger(__name__)

router = APIRouter(tags=["Documents"])

@router.get(
    "/documents",
    response_model=List[DocumentInfo],
    summary="List indexed documents",
    description="Retrieve a list of all documents that have been indexed.",
)
async def list_documents(
    _api_key: Optional[str] = Depends(verify_api_key),
) -> List[DocumentInfo]:
    """List all indexed documents."""
    docs = await redis_manager.get_all_documents()
    return [DocumentInfo(**doc) for doc in docs]

@router.delete(
    "/documents/{doc_id}",
    summary="Delete an indexed document",
    description="Remove a document and all its chunks/images from the index.",
)
async def delete_document(
    doc_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Remove a document and its associated data from Milvus and Redis."""
    logger.info("document_delete_request", doc_id=doc_id)
    
    errors = []
    
    # 1. Delete from Milvus Text Collection
    try:
        if milvus_manager.text_collection:
            milvus_manager.text_collection.delete(f'doc_id == "{doc_id}"')
    except Exception as e:
        errors.append(f"Milvus text deletion failed: {e}")

    # 2. Delete from Milvus Image Collection
    try:
        if milvus_manager.image_collection:
            milvus_manager.image_collection.delete(f'doc_id == "{doc_id}"')
    except Exception as e:
        errors.append(f"Milvus image deletion failed: {e}")

    # 3. Delete from Redis
    try:
        await redis_manager.delete_document(doc_id)
    except Exception as e:
        errors.append(f"Redis deletion failed: {e}")
        
    return {"doc_id": doc_id, "deleted": True, "errors": errors}

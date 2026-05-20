"""
PDF ingestion API endpoints.
"""

from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
import shutil
import tempfile
import os

from app.api.dependencies import verify_api_key
from app.core.logging import get_logger
from app.models.response import IngestResponse, IngestStatusResponse
from app.ingestion.tasks import ingest_pdf_task, celery_app

logger = get_logger(__name__)

router = APIRouter(tags=["Ingestion"])


@router.post(
    "/ingest/pdf",
    response_model=IngestResponse,
    summary="Upload and ingest a PDF manual",
    description="Upload a PDF file for text and image extraction, embedding, and indexing.",
)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    product_id: Optional[str] = Form(None),
    language: str = Form("en"),
    _api_key: Optional[str] = Depends(verify_api_key),
) -> IngestResponse:
    """
    Accept a PDF upload and queue it for async ingestion.
    """
    logger.info(
        "pdf_ingest_request",
        filename=file.filename,
        product_id=product_id,
        language=language,
    )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Save to temporary file
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        os.remove(temp_path)
        logger.error("pdf_save_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Queue Celery task
    task = ingest_pdf_task.delay(
        pdf_path=temp_path,
        filename=file.filename,
        product_id=product_id,
        language=language,
    )

    return IngestResponse(
        job_id=task.id,
        status="queued",
        message=f"PDF '{file.filename}' queued for ingestion.",
    )


@router.get(
    "/ingest/status/{job_id}",
    response_model=IngestStatusResponse,
    summary="Check ingestion job status",
)
async def ingest_status(
    job_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
) -> IngestStatusResponse:
    """Check the status of an async ingestion job."""
    from celery.result import AsyncResult
    task_result = AsyncResult(job_id, app=celery_app)
    
    status = task_result.status.lower()
    errors = []
    total_chunks = 0
    total_images = 0

    if status == "success":
        result_data = task_result.result or {}
        if result_data.get("status") == "completed_with_errors":
            status = "completed_with_errors"
        elif result_data.get("status") == "failed":
            status = "failed"
            
        total_chunks = result_data.get("total_chunks", 0)
        total_images = result_data.get("total_images", 0)
        errors = result_data.get("errors", [])
    elif status == "failure":
        errors = [str(task_result.result)]

    return IngestStatusResponse(
        job_id=job_id,
        status=status,
        progress=100.0 if status in ("success", "failure", "completed_with_errors", "failed") else 0.0,
        total_chunks=total_chunks,
        total_images=total_images,
        errors=errors,
    )

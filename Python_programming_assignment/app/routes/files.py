"""
CSV file upload route.
Validates file type/size, uploads to Supabase Storage, saves metadata, and returns analysis.
Provides endpoints to list and download user files.
"""

import io
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import pandas as pd

from app.config import get_settings
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User
from app.models.file import FileUpload
from app.schemas.file import FileUploadResponse, FileListResponse, FileInfo
from app.supabase_client import supabase

settings = get_settings()

router = APIRouter(prefix="/files", tags=["File Upload"])

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CSV file",
)
async def upload_csv(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload a CSV file for the authenticated user.

    - Validates that the file has a `.csv` extension.
    - Enforces a maximum file size of 5 MB.
    - Uploads the file to Supabase Storage.
    - Saves file metadata to the database.
    - Returns CSV analysis: row/column counts, column names, data types, and missing values.
    """
    # --- File type validation ---
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed.",
        )

    # --- Read file content and enforce size limit ---
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the maximum limit of 5 MB.",
        )

    # --- Load into pandas for analysis ---
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to parse the CSV file. Please check the file format.",
        )

    # --- Upload to Supabase Storage ---
    storage_path = f"{current_user.id}/{file.filename}"
    try:
        supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).upload(
            path=storage_path,
            file=content,
            file_options={"content-type": "text/csv", "upsert": "true"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file to storage: {str(e)}",
        )

    # --- Upsert metadata in database ---
    existing_record = (
        db.query(FileUpload)
        .filter(
            FileUpload.user_id == current_user.id,
            FileUpload.file_name == file.filename,
        )
        .first()
    )
    if existing_record:
        existing_record.storage_path = storage_path
        existing_record.uploaded_at = datetime.now(timezone.utc)
    else:
        file_record = FileUpload(
            user_id=current_user.id,
            file_name=file.filename,
            storage_path=storage_path,
        )
        db.add(file_record)
    db.commit()

    # --- Build analysis response ---
    return FileUploadResponse(
        file_name=file.filename,
        total_rows=len(df),
        total_columns=len(df.columns),
        column_names=df.columns.tolist(),
        data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
        missing_values=df.isnull().sum().to_dict(),
    )


# ---------------------------------------------------------------------------
# List uploaded files
# ---------------------------------------------------------------------------
@router.get(
    "/list",
    response_model=FileListResponse,
    summary="List all files uploaded by the current user",
)
def list_user_files(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return the names, storage paths, and upload timestamps of every file
    that the authenticated user has uploaded or cleaned.
    """
    file_records = (
        db.query(FileUpload)
        .filter(FileUpload.user_id == current_user.id)
        .order_by(FileUpload.uploaded_at.desc())
        .all()
    )

    files = [
        FileInfo(
            file_name=record.file_name,
            #storage_path=record.storage_path,
            uploaded_at=record.uploaded_at,
        )
        for record in file_records
    ]

    return FileListResponse(total_files=len(files), files=files)


# ---------------------------------------------------------------------------
# Download a file
# ---------------------------------------------------------------------------
@router.get(
    "/download/{file_name}",
    summary="Download a file uploaded by the current user",
)
def download_user_file(
    file_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Download a specific file belonging to the authenticated user from
    Supabase Storage and return it as a CSV attachment.
    """
    # --- Find the file record in the database ---
    file_record = (
        db.query(FileUpload)
        .filter(
            FileUpload.user_id == current_user.id,
            FileUpload.file_name == file_name,
        )
        .first()
    )
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_name}' not found for the current user.",
        )

    # --- Download the file from Supabase Storage ---
    try:
        file_bytes = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).download(
            file_record.storage_path
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download file from storage: {str(e)}",
        )

    # --- Stream the file back to the client ---
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{file_name}"'
        },
    )

"""
CSV data cleaning route.
Fetches a previously uploaded CSV, applies a cleaning operation, and uploads the result.
"""

import io
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd

from app.config import get_settings
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User
from app.models.file import FileUpload
from app.schemas.cleaning import CleanRequest, CleanResponse
from app.supabase_client import supabase

settings = get_settings()

router = APIRouter(prefix="/files", tags=["Data Cleaning"])

# Mapping of clean operation IDs to descriptions (for reference)
CLEAN_OPERATIONS = {
    1: "Drop rows with any missing values",
    2: "Fill missing values with column mean",
    3: "Forward fill missing values",
    4: "Backward fill missing values",
    5: "Remove duplicate rows",
}


@router.post(
    "/clean",
    response_model=CleanResponse,
    summary="Clean a previously uploaded CSV file",
)
def clean_csv(
    request: CleanRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Apply a predefined cleaning operation to a user's previously uploaded CSV.

    Operations:
    - 1: Drop all rows containing any missing values.
    - 2: Fill missing values with the respective column mean.
    - 3: Apply forward fill to handle missing values.
    - 4: Apply backward fill to handle missing values.
    - 5: Remove duplicate rows from the dataset.

    The cleaned file is uploaded back to Supabase Storage and the result stats are returned.
    """
    # --- Validate operation ID ---
    if request.clean not in CLEAN_OPERATIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid clean operation. Must be 1-5.",
        )

    # --- Find the file record in the database ---
    file_record = (
        db.query(FileUpload)
        .filter(
            FileUpload.user_id == current_user.id,
            FileUpload.file_name == request.file_name,
        )
        .first()
    )
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{request.file_name}' not found for the current user.",
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

    # --- Load into pandas DataFrame ---
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to parse the stored CSV file.",
        )

    rows_before = len(df)
    nan_filled = 0

    # --- Apply cleaning operation ---
    if request.clean == 1:
        df_cleaned = df.dropna()
    elif request.clean == 2:
        # Fill only numeric columns with mean; leave non-numeric as-is
        numeric_cols = df.select_dtypes(include="number").columns
        df_cleaned = df.copy()
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
            df_cleaned[numeric_cols].mean()
        )
        nan_filled = int(df.isna().sum().sum() - df_cleaned.isna().sum().sum())
    elif request.clean == 3:
        df_cleaned = df.ffill()
        nan_filled = int(df.isna().sum().sum() - df_cleaned.isna().sum().sum())
    elif request.clean == 4:
        df_cleaned = df.bfill()
        nan_filled = int(df.isna().sum().sum() - df_cleaned.isna().sum().sum())
    elif request.clean == 5:
        df_cleaned = df.drop_duplicates()

    rows_after = len(df_cleaned)

    # --- Save cleaned file to Supabase Storage ---
    cleaned_file_name = f"cleaned_{request.file_name}"
    cleaned_storage_path = f"{current_user.id}/{cleaned_file_name}"

    csv_buffer = io.BytesIO()
    df_cleaned.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()

    try:
        supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).upload(
            path=cleaned_storage_path,
            file=csv_bytes,
            file_options={"content-type": "text/csv", "upsert": "true"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload cleaned file to storage: {str(e)}",
        )

    # --- Save cleaned file metadata to database ---
    cleaned_record = FileUpload(
        user_id=current_user.id,
        file_name=cleaned_file_name,
        storage_path=cleaned_storage_path,
    )
    db.add(cleaned_record)
    db.commit()

    return CleanResponse(
        original_file_name=request.file_name,
        cleaned_file_name=cleaned_file_name,
        rows_before=rows_before,
        rows_after=rows_after,
        records_removed_or_modified=abs(rows_before - rows_after),
        nan_filled=nan_filled,
    )

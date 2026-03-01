"""
Pydantic schemas for CSV data cleaning operations.
"""

from pydantic import BaseModel, Field


class CleanRequest(BaseModel):
    """Schema for the data cleaning request body."""

    file_name: str
    clean: int = Field(..., ge=1, le=5, description="Cleaning operation ID (1-5)")


class CleanResponse(BaseModel):
    """Schema for the JSON response returned after cleaning."""

    original_file_name: str
    cleaned_file_name: str
    rows_before: int
    rows_after: int
    records_removed_or_modified: int
    nan_filled: int


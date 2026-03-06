"""
Pydantic schemas for CSV file upload responses.
"""

from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel, field_serializer


class FileUploadResponse(BaseModel):
    """Schema for the JSON response returned after a CSV upload."""

    file_name: str
    total_rows: int
    total_columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]


class FileInfo(BaseModel):
    """Schema representing a single uploaded file entry."""

    file_name: str
    uploaded_at: datetime

    @field_serializer("uploaded_at")
    def format_uploaded_at(self, value: datetime, _info) -> str:
        """Return a human-readable timestamp, e.g. '06 Mar 2026, 09:12 PM'."""
        return value.strftime("%d %b %Y, %I:%M %p")

    class Config:
        from_attributes = True


class FileListResponse(BaseModel):
    """Schema for the JSON response listing a user's files."""

    total_files: int
    files: List[FileInfo]

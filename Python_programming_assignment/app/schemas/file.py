"""
Pydantic schemas for CSV file upload responses.
"""

from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel


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
    #storage_path: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class FileListResponse(BaseModel):
    """Schema for the JSON response listing a user's files."""

    total_files: int
    files: List[FileInfo]

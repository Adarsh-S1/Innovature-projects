"""
Pydantic schemas for CSV file upload responses.
"""

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

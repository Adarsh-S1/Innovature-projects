"""
SQLAlchemy model for the file_uploads table.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Uuid
from app.database import Base


class FileUpload(Base):
    """Represents a CSV file uploaded by a user."""

    __tablename__ = "file_uploads"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        Uuid(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    file_name = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    uploaded_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

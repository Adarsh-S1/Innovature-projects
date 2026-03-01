"""
SQLAlchemy model for the users table.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Uuid
from app.database import Base


class User(Base):
    """Represents a registered user."""

    __tablename__ = "users"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class TokenBlocklist(Base):
    """Represents a blocklisted JWT token for logout functionality."""

    __tablename__ = "token_blocklist"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)

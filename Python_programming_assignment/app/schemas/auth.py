"""
Pydantic schemas for user authentication (registration, login, token response).
"""

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    """Schema for user registration request."""

    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """Schema for user login request."""

    email: EmailStr
    password: str


class Token(BaseModel):
    """Schema for JWT token response."""

    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Schema for user data in responses."""

    id: UUID
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class ForgotPassword(BaseModel):
    """Schema for forgot password request."""

    email: EmailStr


class ResetPassword(BaseModel):
    """Schema for resetting password with token."""

    token: str
    new_password: str


class MessageResponse(BaseModel):
    """Generic schema for returning a message."""

    message: str

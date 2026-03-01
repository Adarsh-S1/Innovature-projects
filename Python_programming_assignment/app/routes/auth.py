"""
Authentication routes: user registration and login.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from datetime import datetime, timezone, timedelta

from app.database import get_db
from app.models.user import User, TokenBlocklist
from app.schemas.auth import (
    UserCreate, UserLogin, Token, UserResponse,
    ForgotPassword, ResetPassword, MessageResponse
)
from app.utils.security import hash_password, verify_password
from app.utils.jwt_handler import create_access_token, decode_access_token
from app.dependencies.auth import get_current_user, oauth2_scheme

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.

    - Validates that the email is not already registered.
    - Hashes the password with bcrypt before saving.
    - Returns the created user profile.
    """
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists.",
        )

    new_user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


@router.post(
    "/login",
    response_model=Token,
    summary="Login and obtain a JWT token",
)
def login(user_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Authenticate a user and issue a JWT access token.

    - Verifies the email exists and the password hash matches.
    - Returns a Bearer token on success.
    """
    user = db.query(User).filter(User.email == user_data.username).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    access_token = create_access_token(data={"sub": user.email})
    return Token(access_token=access_token)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user details",
)
def get_user_me(current_user: User = Depends(get_current_user)):
    """
    Fetch the currently authenticated user's details.
    """
    return current_user


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout a user",
)
def logout(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Logout the user by adding their JWT token to the blocklist.
    """
    # Use the token's expiration date or set a reasonable default for blocklist cleanup
    payload = decode_access_token(token)
    exp = payload.get("exp") if payload else None
    
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else datetime.now(timezone.utc) + timedelta(days=1)

    blocklisted_token = TokenBlocklist(token=token, expires_at=expires_at)
    db.add(blocklisted_token)
    db.commit()

    return MessageResponse(message="Successfully logged out.")


@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    summary="Request a password reset link",
)
def forgot_password(data: ForgotPassword, db: Session = Depends(get_db)):
    """
    Request a password reset token. In a real application, this would send an email.
    For this implementation, the token is returned in the response for testing.
    """
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        # To prevent email enumeration, we return the same message whether user exists or not
        return MessageResponse(message="If an account with that email exists, a password reset token has been generated.")

    # Create a short-lived token (15 mins)
    reset_token = create_access_token(
        data={"sub": user.email, "type": "reset"},
        expires_delta=timedelta(minutes=15)
    )
    
    # Normally, send email here. But for testing, return the token in the message
    return MessageResponse(message=f"Reset token generated: {reset_token}")


@router.post(
    "/reset-password",
    response_model=MessageResponse,
    summary="Reset password using a token",
)
def reset_password(data: ResetPassword, db: Session = Depends(get_db)):
    """
    Reset user password using a valid reset token.
    """
    payload = decode_access_token(data.token)
    if not payload or payload.get("type") != "reset":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token.",
        )
    
    # Ensure token is not blocklisted
    is_blocklisted = db.query(TokenBlocklist).filter(TokenBlocklist.token == data.token).first()
    if is_blocklisted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token has already been used.",
        )

    user_email = payload.get("sub")
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    # Update password
    user.hashed_password = hash_password(data.new_password)
    db.add(user)

    # Blocklist the token so it cannot be reused
    exp = payload.get("exp")
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else datetime.now(timezone.utc) + timedelta(minutes=15)
    blocklisted_token = TokenBlocklist(token=data.token, expires_at=expires_at)
    db.add(blocklisted_token)

    db.commit()

    return MessageResponse(message="Password has been successfully reset.")

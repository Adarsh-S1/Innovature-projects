"""
API dependencies — auth, session management, and rate limiting.
"""

from typing import Optional

from fastapi import Depends, Header, HTTPException, Request

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError, RateLimitError
from app.db.redis_client import redis_manager


async def get_current_settings() -> Settings:
    """Dependency to inject application settings."""
    return get_settings()


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    settings: Settings = Depends(get_current_settings),
) -> Optional[str]:
    """
    Verify the API key if authentication is enabled.
    If no API_KEY is configured, auth is skipped (development mode).
    """
    if not settings.API_KEY:
        return None  # Auth disabled

    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )
    return x_api_key


async def rate_limit_check(
    request: Request,
    settings: Settings = Depends(get_current_settings),
) -> None:
    """
    Sliding-window rate limiter using Redis.
    Identifies clients by IP address.
    """
    client_ip = request.client.host if request.client else "unknown"
    try:
        allowed = await redis_manager.check_rate_limit(client_ip)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {settings.RATE_LIMIT_PER_MINUTE} requests per minute.",
            )
    except RateLimitError:
        raise
    except HTTPException:
        raise
    except Exception:
        # If Redis is down, allow the request (fail-open)
        pass

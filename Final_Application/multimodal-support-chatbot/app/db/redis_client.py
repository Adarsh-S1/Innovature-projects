"""
Redis client — session management, caching, and rate limiting.
"""

import json
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.exceptions import RedisConnectionError
from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisManager:
    """
    Manages Redis connections for session state, caching, and rate limiting.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Establish async connection to Redis."""
        try:
            self._client = aioredis.from_url(
                self._settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Verify connection
            await self._client.ping()
            logger.info(
                "redis_connected",
                host=self._settings.REDIS_HOST,
                port=self._settings.REDIS_PORT,
            )
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            raise RedisConnectionError(f"Failed to connect to Redis: {e}")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            logger.info("redis_disconnected")

    @property
    def client(self) -> aioredis.Redis:
        if not self._client:
            raise RedisConnectionError("Redis client not initialized. Call connect() first.")
        return self._client

    # ── Session Management ───────────────────────────────────────────────

    async def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Store session data with TTL."""
        key = f"session:{session_id}"
        await self.client.setex(
            key,
            self._settings.REDIS_SESSION_TTL_SECONDS,
            json.dumps(data, default=str),
        )
        logger.debug("session_saved", session_id=session_id)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data."""
        key = f"session:{session_id}"
        data = await self.client.get(key)
        if data:
            return json.loads(data)
        return None

    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        key = f"session:{session_id}"
        await self.client.delete(key)
        logger.debug("session_deleted", session_id=session_id)

    # ── Document Metadata ────────────────────────────────────────────────

    async def save_document(self, doc_id: str, data: Dict[str, Any]) -> None:
        """Store document metadata indefinitely."""
        key = f"doc:{doc_id}"
        await self.client.set(key, json.dumps(data, default=str))
        logger.debug("document_saved", doc_id=doc_id)

    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all document metadata."""
        docs = []
        async for key in self.client.scan_iter(match="doc:*"):
            data = await self.client.get(key)
            if data:
                docs.append(json.loads(data))
        return docs
        
    async def delete_document(self, doc_id: str) -> None:
        key = f"doc:{doc_id}"
        await self.client.delete(key)
        logger.debug("document_deleted", doc_id=doc_id)

    # ── Cache ────────────────────────────────────────────────────────────

    async def cache_get(self, key: str) -> Optional[str]:
        """Get a cached value."""
        return await self.client.get(f"cache:{key}")

    async def cache_set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a cached value with optional TTL."""
        ttl = ttl or self._settings.REDIS_CACHE_TTL_SECONDS
        await self.client.setex(f"cache:{key}", ttl, value)

    # ── Rate Limiting ────────────────────────────────────────────────────

    async def check_rate_limit(self, identifier: str) -> bool:
        """
        Simple sliding-window rate limiter.
        Returns True if request is allowed, False if rate-limited.
        """
        key = f"ratelimit:{identifier}"
        current = await self.client.incr(key)
        if current == 1:
            await self.client.expire(key, 60)  # 1-minute window
        return current <= self._settings.RATE_LIMIT_PER_MINUTE

    # ── Health ───────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Check if Redis is reachable."""
        try:
            if self._client:
                await self._client.ping()
                return True
            return False
        except Exception:
            return False


# Singleton instance
redis_manager = RedisManager()

"""
Health check and metrics API endpoints.
"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.milvus_client import milvus_manager
from app.db.minio_client import minio_manager
from app.db.redis_client import redis_manager
from app.models.response import HealthResponse

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health of the application and its dependencies.",
)
async def health_check() -> HealthResponse:
    """Health check endpoint — verifies all dependent services."""
    settings = get_settings()
    services = {}
    overall = "healthy"

    # Check Redis
    try:
        redis_ok = await redis_manager.health_check()
        services["redis"] = "healthy" if redis_ok else "unhealthy"
    except Exception:
        services["redis"] = "unhealthy"

    # Check Milvus
    try:
        milvus_ok = await milvus_manager.health_check()
        services["milvus"] = "healthy" if milvus_ok else "unhealthy"
    except Exception:
        services["milvus"] = "unhealthy"

    # Check MinIO
    try:
        minio_ok = await minio_manager.health_check()
        services["minio"] = "healthy" if minio_ok else "unhealthy"
    except Exception:
        services["minio"] = "unhealthy"

    # Determine overall status
    if all(v == "healthy" for v in services.values()):
        overall = "healthy"
    elif any(v == "unhealthy" for v in services.values()):
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        services=services,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Expose metrics for Prometheus scraping.",
)
async def metrics():
    """Prometheus metrics endpoint — will be implemented with OpenTelemetry."""
    # TODO: Integrate with OpenTelemetry + Prometheus (Part 4)
    return {"status": "metrics_not_yet_configured"}

"""
FastAPI application entrypoint.
Sets up the app, registers routes, middleware, and lifecycle events.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.middleware import setup_middleware
from app.api.v1 import chat, documents, health, ingest
from app.core.config import get_settings
from app.core.exceptions import ChatbotBaseException
from app.core.logging import get_logger, setup_logging
from app.db.milvus_client import milvus_manager
from app.db.minio_client import minio_manager
from app.db.redis_client import redis_manager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Connects to all services on startup, disconnects on shutdown.
    """
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    logger.info(
        "application_starting",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    # ── Startup: connect to services ─────────────────────────────────────
    try:
        await redis_manager.connect()
        logger.info("redis_ready")
    except Exception as e:
        logger.warning("redis_startup_failed", error=str(e))

    try:
        await milvus_manager.connect()
        await milvus_manager.ensure_collections()
        logger.info("milvus_ready")
    except Exception as e:
        logger.warning("milvus_startup_failed", error=str(e))

    try:
        await minio_manager.connect()
        logger.info("minio_ready")
    except Exception as e:
        logger.warning("minio_startup_failed", error=str(e))

    logger.info("application_started")
    yield

    # ── Shutdown: disconnect from services ───────────────────────────────
    logger.info("application_shutting_down")

    await redis_manager.disconnect()
    await milvus_manager.disconnect()

    logger.info("application_stopped")


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Multimodal Intelligent Customer Support Chatbot — "
            "A multi-agent RAG pipeline that retrieves text and images "
            "from technical PDF manuals to answer user queries."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────
    setup_middleware(app)

    # ── Exception Handlers ───────────────────────────────────────────────
    @app.exception_handler(ChatbotBaseException)
    async def chatbot_exception_handler(request: Request, exc: ChatbotBaseException):
        logger.error(
            "chatbot_exception",
            error_code=exc.error_code,
            message=exc.message,
            status_code=exc.status_code,
            context=exc.context,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            error=str(exc),
            path=request.url.path,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred.",
                "detail": str(exc) if settings.DEBUG else None,
            },
        )

    # ── Register API Routers ─────────────────────────────────────────────
    api_prefix = settings.API_PREFIX

    app.include_router(chat.router, prefix=api_prefix)
    app.include_router(ingest.router, prefix=api_prefix)
    app.include_router(documents.router, prefix=api_prefix)
    app.include_router(health.router, prefix=api_prefix)

    # ── Root Endpoint ────────────────────────────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": f"{api_prefix}/health",
        }

    return app


# Create the app instance
app = create_app()

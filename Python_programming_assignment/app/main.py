"""
FastAPI application entry point.
Configures CORS, centralized error handling, lifespan events, and router includes.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.database import engine, Base
from app.routes import auth, files, cleaning


# ---------------------------------------------------------------------------
# Lifespan — create database tables on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create all database tables on application startup."""
    Base.metadata.create_all(bind=engine)
    yield


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FastAPI CSV Manager",
    description="Production-ready API for user authentication, CSV uploads, and data cleaning.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   #Need to change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Centralized error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle all HTTPException responses in a consistent JSON format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "detail": exc.detail},
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=422,
        content={"success": False, "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected server errors."""
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "An unexpected error occurred."},
    )


# ---------------------------------------------------------------------------
# Router includes
# ---------------------------------------------------------------------------
app.include_router(auth.router)
app.include_router(files.router)
app.include_router(cleaning.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def health_check():
    """Root endpoint for health checking."""
    return {"status": "healthy", "message": "FastAPI CSV Manager is running."}

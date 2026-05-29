"""
Chat API endpoints — main chat interface.
Connects to the LangGraph 5-agent pipeline.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.dependencies import rate_limit_check, verify_api_key
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.redis_client import redis_manager
from app.models.request import ChatRequest
from app.models.response import (
    ChatResponse,
    ImageResult,
    SourceReference,
)

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    description="Main chat endpoint. Processes user query through the multi-agent pipeline.",
)
async def chat(
    request: ChatRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
    _rate_limit: None = Depends(rate_limit_check),
) -> ChatResponse:
    """
    Process a user query through the LangGraph agent pipeline.

    1. Route the query (Agent 1 — Context Router)
    2. Perform hybrid text search (Agent 2 — Hybrid Search)
    3. Optionally retrieve images (Agent 3 — Visual Specialist)
    4. Synthesize an answer (Agent 4 — Answer Synthesizer)
    5. Validate quality (Agent 5 — Quality Guard, with loopback)
    """
    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        "chat_request_received",
        session_id=session_id,
        query_length=len(request.query),
        stream=request.stream,
    )

    # ── Load conversation history from Redis ─────────────────────────
    conversation_history: List[Dict[str, Any]] = []
    try:
        session_data = await redis_manager.get_session(session_id)
        if session_data:
            conversation_history = session_data.get("history", [])
    except Exception as e:
        logger.debug("session_load_skipped", error=str(e))

    # ── Build initial agent state ────────────────────────────────────
    initial_state = {
        "user_query": request.query,
        "session_id": session_id,
        # Routing (populated by Agent 1)
        "query_intent": "",
        "needs_image": False,
        "query_keywords": [],
        "query_concepts": [],
        "query_type": "general",
        "reformulated_query": request.query,
        "bm25_weight": 0.4,
        "vector_weight": 0.6,
        # Retrieval (populated by Agents 2 & 3)
        "retrieved_chunks": [],
        "retrieved_images": [],
        "rerank_scores": [],
        # Generation (populated by Agent 4)
        "draft_answer": "",
        "final_answer": "",
        "cited_sources": [],
        # Quality (populated by Agent 5)
        "quality_score": 0.0,
        "quality_issues": [],
        "retry_count": 0,
        # Conversation
        "messages": [],
        "conversation_history": conversation_history,
        # Output
        "response_images": [],
        "response_metadata": {},
    }

    # ── Execute LangGraph pipeline ───────────────────────────────────
    try:
        from app.agents.graph import compiled_graph

        final_state = compiled_graph.invoke(initial_state)

    except Exception as e:
        logger.error("pipeline_execution_failed", error=str(e), exc_info=True)
        return ChatResponse(
            session_id=session_id,
            answer=(
                "I encountered an error processing your request. "
                "Please try again or rephrase your question."
            ),
            images=[],
            sources=[],
            confidence_level="low",
            quality_score=0.0,
            metadata={"error": str(e), "pipeline_connected": True},
        )

    # ── Extract results from final state ─────────────────────────────
    final_answer = final_state.get("final_answer", "")
    quality_score = final_state.get("quality_score", 0.0)
    response_metadata = final_state.get("response_metadata", {})
    confidence_level = response_metadata.get("confidence_level", "medium")

    # ── Build image results ──────────────────────────────────────────
    images = await _build_image_results(final_state.get("response_images", []))

    # ── Build source references ──────────────────────────────────────
    sources = _build_source_references(final_state.get("retrieved_chunks", []))

    # ── Save conversation turn to Redis ──────────────────────────────
    try:
        conversation_history.append({"role": "user", "content": request.query})
        conversation_history.append({"role": "assistant", "content": final_answer})

        # Keep last 20 turns
        conversation_history = conversation_history[-20:]

        await redis_manager.save_session(session_id, {
            "session_id": session_id,
            "history": conversation_history,
        })
    except Exception as e:
        logger.debug("session_save_skipped", error=str(e))

    # ── Compute latency ──────────────────────────────────────────────
    latency_ms = int((time.time() - start_time) * 1000)

    logger.info(
        "chat_response_sent",
        session_id=session_id,
        answer_len=len(final_answer),
        confidence=confidence_level,
        quality_score=round(quality_score, 3),
        latency_ms=latency_ms,
        images=len(images),
        sources=len(sources),
    )

    return ChatResponse(
        session_id=session_id,
        answer=final_answer,
        images=images,
        sources=sources,
        confidence_level=confidence_level,
        quality_score=round(quality_score, 3),
        metadata={
            **response_metadata,
            "latency_ms": latency_ms,
            "pipeline_connected": True,
        },
    )


@router.get(
    "/sessions/{session_id}",
    summary="Get session history",
    description="Retrieve the conversation history for a session.",
)
async def get_session(
    session_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Retrieve conversation history for a given session."""
    try:
        session_data = await redis_manager.get_session(session_id)
        if session_data:
            return {
                "session_id": session_id,
                "history": session_data.get("history", []),
            }
        return {"session_id": session_id, "history": [], "message": "Session not found"}
    except Exception as e:
        logger.warning("session_retrieval_failed", error=str(e))
        return {"session_id": session_id, "history": [], "error": str(e)}


@router.delete(
    "/sessions/{session_id}",
    summary="Clear session",
    description="Delete a session and its conversation history.",
)
async def delete_session(
    session_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Delete a session and clear its history."""
    try:
        await redis_manager.delete_session(session_id)
        return {"session_id": session_id, "deleted": True}
    except Exception as e:
        logger.warning("session_deletion_failed", error=str(e))
        return {"session_id": session_id, "deleted": False, "error": str(e)}


# ── Helper functions ─────────────────────────────────────────────────────


async def _build_image_results(
    response_images: List[Dict[str, Any]],
) -> List[ImageResult]:
    """Convert raw image dicts to ImageResult response models.

    Generates presigned MinIO URLs so the frontend can securely
    display images without direct access to the storage backend.
    """
    from app.db.minio_client import minio_manager

    results = []
    for img in response_images:
        try:
            # Generate presigned URLs for secure frontend access
            storage_url = img.get("storage_url", "")
            thumbnail_url = img.get("thumbnail_url", "")

            try:
                presigned_url = await minio_manager.get_presigned_url(storage_url) if storage_url else ""
                presigned_thumb = await minio_manager.get_presigned_url(thumbnail_url) if thumbnail_url else ""
            except Exception:
                logger.warning("presigned_url_failed", storage_url=storage_url)
                presigned_url = storage_url
                presigned_thumb = thumbnail_url

            results.append(ImageResult(
                image_id=img.get("image_id", ""),
                url=presigned_url,
                thumbnail_url=presigned_thumb,
                caption=img.get("caption", ""),
                relevance_score=img.get("relevance_score", 0.0),
                image_type=img.get("image_type", "other"),
                source=f"{img.get('source_file', 'Unknown')}, Page {img.get('page_number', '?')}",
            ))
        except Exception:
            continue
    return results


def _build_source_references(
    chunks: List[Dict[str, Any]],
) -> List[SourceReference]:
    """Convert retrieved chunks to SourceReference citations."""
    sources = []
    seen = set()

    for chunk in chunks:
        doc_id = chunk.get("doc_id", "")
        source_file = chunk.get("source_file", "Unknown")

        key = f"{doc_id}:{chunk.get('page_start')}"
        if key in seen:
            continue
        seen.add(key)

        sources.append(SourceReference(
            doc_id=doc_id,
            source_file=source_file,
            section_path=chunk.get("section_path", []),
            page_start=chunk.get("page_start", 0),
            page_end=chunk.get("page_end", 0),
            relevance_score=chunk.get("final_score", 0.0),
        ))

    return sources

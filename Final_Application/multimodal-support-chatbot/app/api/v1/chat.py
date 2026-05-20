"""
Chat API endpoints — main chat interface.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends

from app.api.dependencies import rate_limit_check, verify_api_key
from app.core.logging import get_logger
from app.models.request import ChatRequest
from app.models.response import ChatResponse

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

    This endpoint will:
    1. Route the query (Agent 1)
    2. Perform hybrid text search (Agent 2)
    3. Optionally retrieve images (Agent 3)
    4. Synthesize an answer (Agent 4)
    5. Validate quality (Agent 5)
    """
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        "chat_request_received",
        session_id=session_id,
        query_length=len(request.query),
        stream=request.stream,
    )

    # TODO: Invoke LangGraph pipeline (Part 4)
    # For now, return a stub response
    return ChatResponse(
        session_id=session_id,
        answer="The multi-agent pipeline is not yet connected. This is a stub response from Part 1 setup.",
        images=[],
        sources=[],
        confidence_level="low",
        quality_score=0.0,
        metadata={"status": "stub", "pipeline_connected": False},
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
    # TODO: Implement in Part 4
    return {"session_id": session_id, "history": [], "message": "Not yet implemented"}


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
    # TODO: Implement in Part 4
    return {"session_id": session_id, "deleted": True}

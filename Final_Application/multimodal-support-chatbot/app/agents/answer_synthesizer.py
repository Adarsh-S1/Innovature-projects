"""
Agent 4: Answer Synthesizer — Generates final answer from retrieved context.
Stub implementation; full logic will be added in Part 4.
"""

from app.agents.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)


async def answer_synthesizer(state: AgentState) -> dict:
    """
    Compose a technically accurate, well-structured answer.

    Responsibilities:
    - Use retrieved text chunks as context
    - Include in-text references to images
    - Format response in markdown
    - Cite source sections and page numbers

    Will be fully implemented in Part 4.
    """
    logger.info(
        "answer_synthesizer_invoked",
        num_chunks=len(state.get("retrieved_chunks", [])),
        num_images=len(state.get("retrieved_images", [])),
    )
    return {
        "draft_answer": "Stub answer — will be replaced in Part 4.",
        "cited_sources": [],
    }

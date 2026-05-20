"""
Agent 5: Quality Guard — Scores answer quality and triggers loopback if needed.
Stub implementation; full logic will be added in Part 4.
"""

from app.agents.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)


async def quality_guard(state: AgentState) -> dict:
    """
    Score the synthesized answer on multiple quality dimensions.

    Responsibilities:
    - Score groundedness, completeness, conciseness, accuracy, image relevance
    - Detect hallucinations, vagueness, unanswered questions
    - Trigger loopback to Agent 2 if quality below threshold
    - Approve final answer and finalize response structure

    Will be fully implemented in Part 4.
    """
    retry_count = state.get("retry_count", 0)
    logger.info(
        "quality_guard_invoked",
        retry_count=retry_count,
    )
    return {
        "quality_score": 0.85,
        "quality_issues": [],
        "final_answer": state.get("draft_answer", ""),
        "retry_count": retry_count,
        "response_images": state.get("retrieved_images", []),
        "response_metadata": {
            "query_intent": state.get("query_intent"),
            "retry_count": retry_count,
        },
    }

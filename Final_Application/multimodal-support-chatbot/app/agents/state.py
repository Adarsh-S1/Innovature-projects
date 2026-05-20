"""
AgentState TypedDict — shared state passed through all LangGraph nodes.
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state schema for the 5-agent LangGraph pipeline.

    This state is created at the start of each request and flows
    through: Context Router → Hybrid Search → Visual Specialist
    → Answer Synthesizer → Quality Guard (with possible loopback).
    """

    # ── Input ────────────────────────────────────────────────────────────
    user_query: str
    session_id: str

    # ── Routing (Agent 1 output) ─────────────────────────────────────────
    query_intent: str              # "text_only" | "image_only" | "multimodal"
    needs_image: bool
    query_keywords: List[str]      # Extracted keywords for BM25
    query_concepts: List[str]      # Extracted concepts for vector search
    query_type: str                # "how_to" | "troubleshoot" | "locate_component" | ...
    reformulated_query: str        # Context-enriched version of the query
    bm25_weight: float             # Dynamic BM25 weight (set by router)
    vector_weight: float           # Dynamic vector weight (set by router)

    # ── Retrieval Results (Agent 2 output) ───────────────────────────────
    retrieved_chunks: List[Dict[str, Any]]   # Top-K text chunks with scores
    retrieved_images: List[Dict[str, Any]]   # Top-K images with scores
    rerank_scores: List[float]

    # ── Generation (Agent 4 output) ──────────────────────────────────────
    draft_answer: str
    final_answer: str
    cited_sources: List[str]

    # ── Quality Control (Agent 5 output) ─────────────────────────────────
    quality_score: float
    quality_issues: List[str]
    retry_count: int               # Max retries = 2

    # ── Conversation ─────────────────────────────────────────────────────
    messages: Annotated[List, add_messages]
    conversation_history: List[Dict[str, Any]]

    # ── Output ───────────────────────────────────────────────────────────
    response_images: List[Dict[str, Any]]    # Final images to return
    response_metadata: Dict[str, Any]

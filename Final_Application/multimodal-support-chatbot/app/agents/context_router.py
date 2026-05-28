"""
Agent 1: Context Router — Parses query intent and routes to the appropriate pipeline path.

Uses GPT-4o with structured output to classify the query and extract
keywords, concepts, error codes. Falls back to rule-based classification
when the LLM is unavailable.
"""

import json
import re
from typing import Any, Dict, List

from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Rule-based keyword patterns ──────────────────────────────────────────

_IMAGE_PATTERNS = re.compile(
    r'\b(where\s+is|show\s+me|diagram|layout|photo|image|picture|schematic|'
    r'what\s+does\s+\w+\s+look\s+like|locate|location\s+of|visual)\b',
    re.IGNORECASE,
)

_HOW_TO_PATTERNS = re.compile(
    r'\b(how\s+to|steps?\s+to|procedure|install|remove|replace|configure|setup|'
    r'troubleshoot|fix|repair|guide|instructions?)\b',
    re.IGNORECASE,
)

_ERROR_CODE_PATTERN = re.compile(
    r'\b([A-Z]{1,3}[\-_]?\d{2,5}[A-Za-z]?)\b'
)

_SPEC_PATTERNS = re.compile(
    r'\b(specification|specs?|dimension|weight|capacity|voltage|wattage|'
    r'resolution|speed|frequency|temperature|rating|model\s+number)\b',
    re.IGNORECASE,
)

_LOCATE_PATTERNS = re.compile(
    r'\b(where|locate|find|position|location|slot|port|connector|socket)\b',
    re.IGNORECASE,
)

# System prompt for GPT-4o structured classification
_ROUTER_SYSTEM_PROMPT = """You are a query classification engine for a technical support chatbot.
Given a user query and optional conversation history, classify the query and extract structured parameters.

Respond ONLY with valid JSON matching this exact schema:
{
  "intent": "text_only | image_only | multimodal",
  "needs_image": true/false,
  "detected_components": ["component1", "component2"],
  "query_type": "how_to | troubleshoot | locate_component | specification | general",
  "extracted_error_codes": ["E-47"],
  "is_followup": true/false,
  "reformulated_query": "cleaned, context-enriched version of the query",
  "query_keywords": ["keyword1", "keyword2"],
  "query_concepts": ["concept1", "concept2"]
}

Classification rules:
- "where is", "show me", "diagram of", "what does X look like" → intent=multimodal, needs_image=true
- "how to", "steps to", "procedure for" → intent=text_only (set needs_image=true if physical components mentioned)
- Queries with error codes → intent=text_only, query_type=troubleshoot
- "specifications", "dimensions", "capacity" → intent=text_only, query_type=specification
- If the query references a previous answer or uses pronouns like "it", "that" → is_followup=true
- reformulated_query should expand abbreviations and resolve pronouns using conversation history"""


async def context_router(state: AgentState) -> dict:
    """
    Agent 1: Parse user query for intent classification.

    Attempts GPT-4o structured classification first. Falls back to
    rule-based classification if the LLM is unavailable.
    """
    query = state.get("user_query", "")
    session_id = state.get("session_id", "")
    history = state.get("conversation_history", [])

    logger.info(
        "context_router_invoked",
        query=query,
        session_id=session_id,
        history_turns=len(history),
    )

    # Attempt LLM-based classification
    result = await _llm_classify(query, history)

    if result is None:
        # Fallback to rule-based
        result = _rule_based_classify(query, history)

    logger.info(
        "context_router_completed",
        intent=result.get("query_intent"),
        needs_image=result.get("needs_image"),
        query_type=result.get("query_type"),
    )

    return result


async def _llm_classify(
    query: str, history: List[Dict[str, Any]]
) -> dict | None:
    """Use Groq LLM to classify the query. Returns None on failure."""
    settings = get_settings()

    try:
        from app.core.shared import get_async_groq_client, strip_markdown_fences

        client = get_async_groq_client()

        # Build conversation context
        history_text = ""
        if history:
            last_turns = history[-6:]  # Last 3 exchanges
            history_text = "\n".join(
                f"{t.get('role', 'user')}: {t.get('content', '')}"
                for t in last_turns
            )

        user_content = f"User query: {query}"
        if history_text:
            user_content = f"Conversation history:\n{history_text}\n\n{user_content}"

        response = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()
        content = strip_markdown_fences(content)

        parsed = json.loads(content)

        # Map to agent state keys
        intent = parsed.get("intent", "multimodal")
        query_type = parsed.get("query_type", "general")

        # Dynamic weights based on query type
        bm25_w, vec_w = _get_dynamic_weights(query_type)

        return {
            "query_intent": intent,
            "needs_image": parsed.get("needs_image", True),
            "query_keywords": parsed.get("query_keywords", []),
            "query_concepts": parsed.get("query_concepts", []),
            "query_type": query_type,
            "reformulated_query": parsed.get("reformulated_query", query),
            "bm25_weight": bm25_w,
            "vector_weight": vec_w,
        }

    except Exception as e:
        logger.warning("llm_classification_failed", error=str(e))
        return None


def _rule_based_classify(
    query: str, history: List[Dict[str, Any]]
) -> dict:
    """Fallback rule-based classification using regex patterns."""

    # Detect intent
    needs_image = bool(_IMAGE_PATTERNS.search(query))

    # Detect query type
    error_codes = _ERROR_CODE_PATTERN.findall(query)
    if error_codes:
        query_type = "troubleshoot"
    elif _HOW_TO_PATTERNS.search(query):
        query_type = "how_to"
    elif _LOCATE_PATTERNS.search(query):
        query_type = "locate_component"
        needs_image = True  # Location queries benefit from visuals
    elif _SPEC_PATTERNS.search(query):
        query_type = "specification"
    else:
        query_type = "general"

    # Determine intent
    if needs_image and query_type in ("locate_component",):
        intent = "multimodal"
    elif needs_image:
        intent = "multimodal"
    else:
        intent = "text_only"

    # Extract keywords (simple: all meaningful words)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "shall", "would", "could", "should", "may", "might", "can",
                  "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "as", "into", "about", "it", "its", "this", "that", "which",
                  "what", "how", "where", "when", "who", "i", "me", "my"}
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Check if follow-up
    is_followup = bool(
        re.search(r'\b(it|that|this|those|them|previous|above|earlier)\b', query, re.IGNORECASE)
        and history
    )

    # Reformulate query
    reformulated = query
    if is_followup and history:
        last_query = ""
        for turn in reversed(history):
            if turn.get("role") == "user":
                last_query = turn.get("content", "")
                break
        if last_query:
            reformulated = f"{last_query} — follow-up: {query}"

    # Dynamic weights
    bm25_w, vec_w = _get_dynamic_weights(query_type)

    return {
        "query_intent": intent,
        "needs_image": needs_image,
        "query_keywords": keywords + error_codes,
        "query_concepts": keywords[:5],
        "query_type": query_type,
        "reformulated_query": reformulated,
        "bm25_weight": bm25_w,
        "vector_weight": vec_w,
    }


def _get_dynamic_weights(query_type: str) -> tuple:
    """Return (bm25_weight, vector_weight) based on query type.

    Delegates to the canonical _WEIGHT_TABLE in hybrid_searcher
    to avoid maintaining duplicate weight tables.
    """
    from app.retrieval.hybrid_searcher import _WEIGHT_TABLE
    from app.models.domain import QueryType

    try:
        qt = QueryType(query_type)
        return _WEIGHT_TABLE.get(qt, (0.40, 0.60))
    except ValueError:
        return (0.40, 0.60)

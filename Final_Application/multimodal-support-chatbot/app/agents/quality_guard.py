"""
Agent 5: Quality Guard — Scores answer quality on 5 dimensions
and triggers loopback if quality is below threshold.

Quality Dimensions (0–1 each):
  1. Groundedness — every claim supported by retrieved context?
  2. Completeness — all parts of the query addressed?
  3. Conciseness  — free of unnecessary padding?
  4. Technical Accuracy — component names, codes, procedures correct?
  5. Image Relevance — fetched image actually relevant to the text?

Decision Logic:
  composite ≥ 0.75 → APPROVE
  composite < 0.75 AND retries < 2 → LOOPBACK (enrich query, re-run search)
  composite < 0.75 AND retries ≥ 2 → APPROVE with LOW_CONFIDENCE flag
"""

import json
import re
from typing import Any, Dict, List

from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_QUALITY_SYSTEM_PROMPT = """You are a quality assurance reviewer for a technical support chatbot.
Given a user query, retrieved context passages, and a draft answer, score the answer on 5 dimensions.

Score each dimension from 0.0 to 1.0:
1. groundedness: Is every factual claim in the answer supported by the provided context?
2. completeness: Does the answer address ALL parts of the user's question?
3. conciseness: Is the answer free of unnecessary padding or redundancy?
4. technical_accuracy: Are component names, error codes, procedures technically correct?
5. image_relevance: If images are referenced, are they actually relevant? (1.0 if no images)

Also list any specific quality issues found.

Respond ONLY with valid JSON:
{
  "groundedness": 0.9,
  "completeness": 0.85,
  "conciseness": 0.9,
  "technical_accuracy": 0.95,
  "image_relevance": 1.0,
  "issues": ["issue description 1", "issue description 2"]
}"""


async def quality_guard(state: AgentState) -> dict:
    """
    Agent 5: Score answer quality and decide APPROVE or LOOPBACK.

    Combines rule-based checks with optional LLM-as-judge scoring.
    """
    draft_answer = state.get("draft_answer", "")
    query = state.get("user_query", "")
    chunks = state.get("retrieved_chunks", [])
    images = state.get("retrieved_images", [])
    retry_count = state.get("retry_count", 0)

    logger.info(
        "quality_guard_invoked",
        answer_len=len(draft_answer),
        retry_count=retry_count,
        num_chunks=len(chunks),
    )

    # ── Rule-based quality checks ────────────────────────────────────
    rule_scores, rule_issues = _rule_based_quality(
        draft_answer, query, chunks, images
    )

    # ── Optional: LLM-based quality scoring ──────────────────────────
    llm_scores = await _llm_quality_check(draft_answer, query, chunks)

    # ── Merge scores (prefer LLM if available, blend otherwise) ──────
    if llm_scores:
        # Blend: 60% LLM + 40% rule-based
        dimensions = ["groundedness", "completeness", "conciseness",
                       "technical_accuracy", "image_relevance"]
        final_scores = {}
        for dim in dimensions:
            llm_val = llm_scores.get(dim, 0.8)
            rule_val = rule_scores.get(dim, 0.8)
            final_scores[dim] = 0.6 * llm_val + 0.4 * rule_val

        issues = rule_issues + llm_scores.get("issues", [])
    else:
        final_scores = rule_scores
        issues = rule_issues

    # ── Composite score (weighted average) ───────────────────────────
    weights = {
        "groundedness": 0.30,
        "completeness": 0.25,
        "conciseness": 0.10,
        "technical_accuracy": 0.25,
        "image_relevance": 0.10,
    }

    composite = sum(
        final_scores.get(dim, 0.8) * w
        for dim, w in weights.items()
    )

    # ── Decision: approve or loopback ────────────────────────────────
    settings = get_settings()
    quality_threshold = settings.QUALITY_THRESHOLD

    if composite >= quality_threshold:
        confidence = "high" if composite >= 0.9 else "medium"
        final_answer = draft_answer
    elif retry_count < settings.MAX_RETRIES:
        # Loopback — increment retry, re-run search
        confidence = "low"
        final_answer = draft_answer  # Will be overwritten on next pass
        retry_count += 1
        logger.info(
            "quality_guard_loopback",
            composite=composite,
            retry_count=retry_count,
        )
    else:
        # Max retries reached — approve with low confidence + disclaimer
        confidence = "low"
        final_answer = (
            draft_answer + "\n\n---\n"
            "*⚠️ This response may have lower accuracy than usual. "
            "Please verify critical information against the original documentation.*"
        )

    # ── Build response images ────────────────────────────────────────
    response_images = _build_response_images(images)

    logger.info(
        "quality_guard_completed",
        composite_score=round(composite, 3),
        confidence=confidence,
        retry_count=retry_count,
        issues=len(issues),
    )

    return {
        "quality_score": composite,
        "quality_issues": issues[:10],  # Cap at 10
        "final_answer": final_answer,
        "retry_count": retry_count,
        "response_images": response_images,
        "response_metadata": {
            "query_intent": state.get("query_intent", ""),
            "query_type": state.get("query_type", ""),
            "confidence_level": confidence,
            "quality_dimensions": final_scores,
            "retry_count": retry_count,
            "num_sources": len(chunks),
            "num_images": len(images),
        },
    }


def _rule_based_quality(
    answer: str,
    query: str,
    chunks: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
) -> tuple:
    """
    Rule-based quality scoring. Returns (scores_dict, issues_list).
    """
    issues = []
    scores = {}

    # ── Groundedness: check overlap between answer and context ────
    if chunks:
        chunk_texts = " ".join(c.get("text", "") for c in chunks).lower()
        answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        context_words = set(re.findall(r'\b\w{4,}\b', chunk_texts))

        if answer_words:
            overlap_ratio = len(answer_words & context_words) / len(answer_words)
            scores["groundedness"] = min(overlap_ratio * 1.5, 1.0)
        else:
            scores["groundedness"] = 0.5

        if scores["groundedness"] < 0.5:
            issues.append("Low groundedness: answer may contain unsupported claims")
    else:
        scores["groundedness"] = 0.3
        issues.append("No context chunks available for groundedness check")

    # ── Completeness: check query terms in answer ────────────────
    query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
    stop_words = {"the", "how", "what", "where", "when", "does", "can", "is",
                  "are", "was", "for", "and", "not", "with"}
    query_terms -= stop_words

    if query_terms:
        answer_lower = answer.lower()
        covered = sum(1 for t in query_terms if t in answer_lower)
        scores["completeness"] = covered / len(query_terms)
    else:
        scores["completeness"] = 0.8

    if scores["completeness"] < 0.5:
        issues.append("Answer may not fully address the user's question")

    # ── Conciseness ──────────────────────────────────────────────
    word_count = len(answer.split())
    if word_count > 600:
        scores["conciseness"] = 0.5
        issues.append("Answer is unusually long (>600 words)")
    elif word_count < 10:
        scores["conciseness"] = 0.4
        issues.append("Answer is too short to be useful")
    else:
        scores["conciseness"] = min(1.0, 400 / max(word_count, 1))

    # ── Technical accuracy (basic heuristic) ─────────────────────
    # Check if "I don't know" or hedging language is used
    hedge_count = len(re.findall(
        r'\b(might|maybe|perhaps|possibly|not sure|uncertain)\b',
        answer, re.IGNORECASE
    ))
    scores["technical_accuracy"] = max(0.5, 1.0 - hedge_count * 0.15)

    if hedge_count > 2:
        issues.append("Answer contains excessive hedging language")

    # ── Image relevance ──────────────────────────────────────────
    if images:
        # Check if answer references images
        image_refs = bool(re.search(
            r'\b(diagram|image|figure|shown|illustrated|picture|photo)\b',
            answer, re.IGNORECASE
        ))
        avg_score = sum(
            img.get("composite_score", 0) for img in images
        ) / len(images)

        scores["image_relevance"] = 0.7 if image_refs else 0.4
        scores["image_relevance"] = min(1.0, scores["image_relevance"] + avg_score * 0.3)

        if avg_score < 0.3:
            issues.append("Retrieved images have low relevance scores")
    else:
        scores["image_relevance"] = 1.0  # No images needed = no penalty

    return scores, issues


async def _llm_quality_check(
    answer: str,
    query: str,
    chunks: List[Dict[str, Any]],
) -> dict | None:
    """Optional LLM-based quality scoring using GPT-4o-mini."""
    settings = get_settings()

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            max_retries=1,
            timeout=10,
        )

        context = "\n".join(
            c.get("text", "")[:300] for c in chunks[:5]
        )

        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use mini for cost efficiency
            messages=[
                {"role": "system", "content": _QUALITY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Context:\n{context}\n\n"
                        f"Draft Answer:\n{answer}"
                    ),
                },
            ],
            max_tokens=300,
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]

        return json.loads(content)

    except Exception as e:
        logger.debug("llm_quality_check_skipped", error=str(e))
        return None


def _build_response_images(
    images: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build final response image payload with presigned URLs."""
    response = []

    for img in images:
        score = img.get("composite_score", 0)
        # Drop images below 0.55 relevance threshold (from spec)
        if score < 0.55:
            continue

        response.append({
            "image_id": img.get("image_id", ""),
            "url": img.get("storage_url", ""),
            "thumbnail_url": img.get("thumbnail_url", ""),
            "caption": img.get("caption", ""),
            "relevance_score": round(score, 3),
            "image_type": img.get("image_type", "other"),
            "source": f"{img.get('source_file', 'Unknown')}, Page {img.get('page_number', '?')}",
        })

    return response

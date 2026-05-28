"""
Agent 4: Answer Synthesizer — Generates a technically accurate, well-structured
answer using GPT-4o from retrieved text chunks and images.

Includes in-text image references, markdown formatting, and source citations.
Falls back to a context-summary answer if the LLM is unavailable.
"""

import json
from typing import Any, Dict, List

from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = """You are a senior technical support specialist.
Using ONLY the provided context passages, answer the user's question.

Rules:
1. Be precise and use technical terminology from the source material
2. Structure your answer with headers if multi-step
3. If an image is provided, reference it naturally: "As shown in the diagram..."
4. Cite your sources: [Source: Manual X, Section Y, Page Z]
5. If the context is insufficient, clearly state what is unknown
6. Do not hallucinate or add information beyond the retrieved context
7. Keep answers under 400 words unless a procedure demands more
8. Use markdown formatting (headers, bold, lists) for readability"""


async def answer_synthesizer(state: AgentState) -> dict:
    """
    Agent 4: Compose a technically accurate, well-structured answer.

    Uses GPT-4o with the retrieved chunks as context and image references.
    Falls back to a direct context summary on LLM failure.
    """
    query = state.get("user_query", "")
    chunks = state.get("retrieved_chunks", [])
    images = state.get("retrieved_images", [])
    history = state.get("conversation_history", [])

    logger.info(
        "answer_synthesizer_invoked",
        query_len=len(query),
        num_chunks=len(chunks),
        num_images=len(images),
    )

    # ── Handle no context case ───────────────────────────────────────
    if not chunks:
        no_context_answer = (
            "I wasn't able to find relevant information in the indexed documents "
            "to answer your question. Please try rephrasing your query, or ensure "
            "the relevant PDF manuals have been ingested into the system."
        )
        return {
            "draft_answer": no_context_answer,
            "cited_sources": [],
        }

    # ── Build context prompt ─────────────────────────────────────────
    context_text = _build_context_text(chunks)
    image_text = _build_image_text(images)

    # ── Attempt LLM synthesis ────────────────────────────────────────
    draft_answer = await _llm_synthesize(query, context_text, image_text, history)

    if draft_answer is None:
        # Fallback: summarize context directly
        draft_answer = _fallback_synthesis(query, chunks, images)

    # ── Extract cited sources ────────────────────────────────────────
    cited_sources = _extract_sources(chunks)

    logger.info(
        "answer_synthesizer_completed",
        answer_len=len(draft_answer),
        sources=len(cited_sources),
    )

    return {
        "draft_answer": draft_answer,
        "cited_sources": cited_sources,
    }


def _build_context_text(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as numbered context passages."""
    passages = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file", "Unknown")
        pages = f"Pages {chunk.get('page_start', '?')}-{chunk.get('page_end', '?')}"
        section = " > ".join(chunk.get("section_path", []))
        section_str = f", Section: {section}" if section else ""

        passages.append(
            f"[Passage {i}] (Source: {source}, {pages}{section_str})\n"
            f"{chunk.get('text', '')}"
        )

    return "\n\n---\n\n".join(passages)


def _build_image_text(images: List[Dict[str, Any]]) -> str:
    """Format retrieved images as reference descriptions."""
    if not images:
        return "No images are available for this query."

    parts = []
    for i, img in enumerate(images, 1):
        parts.append(
            f"[Image {i}]: {img.get('caption', 'No caption')} "
            f"(Type: {img.get('image_type', 'unknown')}, "
            f"Page: {img.get('page_number', '?')}, "
            f"Score: {img.get('composite_score', 0):.2f})"
        )

    return "\n".join(parts)


async def _llm_synthesize(
    query: str,
    context_text: str,
    image_text: str,
    history: List[Dict[str, Any]],
) -> str | None:
    """Use Groq LLM to synthesize the final answer."""
    settings = get_settings()

    try:
        from app.core.shared import get_async_groq_client

        client = get_async_groq_client()

        # Include last few turns for continuity
        history_messages = []
        if history:
            for turn in history[-6:]:
                role = turn.get("role", "user")
                if role in ("user", "assistant"):
                    history_messages.append({
                        "role": role,
                        "content": turn.get("content", ""),
                    })

        user_content = (
            f"## Context Passages\n\n{context_text}\n\n"
            f"## Available Images\n\n{image_text}\n\n"
            f"## User Question\n\n{query}"
        )

        messages = [
            {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
            *history_messages,
            {"role": "user", "content": user_content},
        ]

        response = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            max_tokens=1500,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning("llm_synthesis_failed", error=str(e))
        return None


def _fallback_synthesis(
    query: str,
    chunks: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
) -> str:
    """Fallback: build answer directly from top chunks without LLM."""
    parts = [f"**Based on the available documentation for: \"{query}\"**\n"]

    for i, chunk in enumerate(chunks[:3], 1):
        source = chunk.get("source_file", "Unknown")
        page = chunk.get("page_start", "?")
        text = chunk.get("text", "")[:500]
        parts.append(
            f"**Passage {i}** *(Source: {source}, Page {page})*\n\n{text}\n"
        )

    if images:
        parts.append("\n**Related Images:**")
        for img in images[:2]:
            parts.append(
                f"- {img.get('caption', 'Image')} "
                f"(Page {img.get('page_number', '?')})"
            )

    parts.append(
        "\n\n*Note: This is a direct context excerpt. "
        "Full LLM-synthesized answers require a valid OpenAI API key.*"
    )

    return "\n".join(parts)


def _extract_sources(chunks: List[Dict[str, Any]]) -> List[str]:
    """Extract unique source citations from chunks."""
    sources = []
    seen = set()

    for chunk in chunks:
        source = chunk.get("source_file", "Unknown")
        page_s = chunk.get("page_start", 0)
        page_e = chunk.get("page_end", 0)
        section = " > ".join(chunk.get("section_path", []))

        citation = f"{source}, Pages {page_s}-{page_e}"
        if section:
            citation += f", Section: {section}"

        if citation not in seen:
            seen.add(citation)
            sources.append(citation)

    return sources

"""
Unit tests for LangGraph agents — focused on testable logic
that doesn't require live LLM or infrastructure connections.

Tests cover:
- Context Router: rule-based classification logic
- Answer Synthesizer: context building, fallback synthesis, source extraction
- Quality Guard: rule-based scoring dimensions
"""

import pytest

# ── Context Router Tests ─────────────────────────────────────────────────

from app.agents.context_router import (
    _rule_based_classify,
    _get_dynamic_weights,
)


class TestRuleBasedClassification:
    """Tests for the rule-based fallback classifier in context_router."""

    def test_how_to_query(self):
        """'how to' queries should be classified as how_to."""
        result = _rule_based_classify("How to install RAM?", [])
        assert result["query_type"] == "how_to"

    def test_troubleshoot_error_code(self):
        """Queries with error codes should be classified as troubleshoot."""
        result = _rule_based_classify("What does error E-47 mean?", [])
        assert result["query_type"] == "troubleshoot"

    def test_locate_component_query(self):
        """'where is' queries should flag needs_image and locate_component."""
        result = _rule_based_classify("Where is the RAM slot?", [])
        assert result["needs_image"] is True
        assert result["query_type"] == "locate_component"

    def test_specification_query(self):
        """'specification' queries should be classified correctly."""
        result = _rule_based_classify(
            "What are the voltage specs for this board?", []
        )
        assert result["query_type"] == "specification"

    def test_general_query(self):
        """Generic queries should fall through to 'general'."""
        result = _rule_based_classify(
            "Tell me about the product features", []
        )
        assert result["query_type"] == "general"

    def test_diagram_triggers_multimodal(self):
        """Queries mentioning 'diagram' should be multimodal."""
        result = _rule_based_classify("Show me the wiring diagram", [])
        assert result["query_intent"] == "multimodal"
        assert result["needs_image"] is True

    def test_keywords_extracted(self):
        """Keywords should be extracted (stop words removed)."""
        result = _rule_based_classify("How to install the RAM module?", [])
        assert "install" in result["query_keywords"]
        assert "ram" in result["query_keywords"]
        assert "the" not in result["query_keywords"]

    def test_followup_detection(self):
        """Follow-up queries should be detected when history exists."""
        history = [
            {"role": "user", "content": "What is the CPU socket type?"},
            {"role": "assistant", "content": "It's an LGA 1700 socket."},
        ]
        result = _rule_based_classify("What about it?", history)
        assert result["reformulated_query"] != "What about it?"

    def test_no_followup_without_history(self):
        """Follow-up should not be detected without conversation history."""
        result = _rule_based_classify("Tell me about it", [])
        assert result["reformulated_query"] == "Tell me about it"

    def test_result_contains_required_keys(self):
        """All required state keys should be present in the result."""
        result = _rule_based_classify("Test query", [])
        required_keys = {
            "query_intent", "needs_image", "query_keywords",
            "query_concepts", "query_type", "reformulated_query",
            "bm25_weight", "vector_weight",
        }
        assert required_keys.issubset(result.keys())


class TestDynamicWeights:
    """Tests for _get_dynamic_weights function."""

    def test_troubleshoot_weights_bm25_heavy(self):
        """Troubleshoot should be BM25-heavy."""
        bm25_w, vec_w = _get_dynamic_weights("troubleshoot")
        assert bm25_w > vec_w

    def test_how_to_weights_vector_heavy(self):
        """How-to should be vector-heavy."""
        bm25_w, vec_w = _get_dynamic_weights("how_to")
        assert vec_w > bm25_w

    def test_unknown_type_returns_default(self):
        """Unknown query type should return default weights."""
        bm25_w, vec_w = _get_dynamic_weights("unknown_type")
        assert (bm25_w, vec_w) == (0.40, 0.60)

    def test_all_weights_sum_to_one(self):
        """All known query types should have weights summing to ~1.0."""
        for qt in ["troubleshoot", "how_to", "locate_component",
                    "specification", "general"]:
            bm25_w, vec_w = _get_dynamic_weights(qt)
            assert abs(bm25_w + vec_w - 1.0) < 0.01


# ── Answer Synthesizer Tests ─────────────────────────────────────────────

from app.agents.answer_synthesizer import (
    _build_context_text,
    _build_image_text,
    _fallback_synthesis,
    _extract_sources,
)


class TestBuildContextText:
    """Tests for context text formatting."""

    def test_formats_passages_with_source(self):
        """Should include source file and page numbers."""
        chunks = [
            {
                "source_file": "manual.pdf",
                "page_start": 5,
                "page_end": 7,
                "section_path": ["Hardware", "RAM"],
                "text": "The RAM slot supports DDR4.",
            }
        ]
        result = _build_context_text(chunks)
        assert "manual.pdf" in result
        assert "Pages 5-7" in result
        assert "RAM slot supports DDR4" in result

    def test_empty_chunks_returns_empty_string(self):
        """Empty chunks should produce empty context."""
        result = _build_context_text([])
        assert result == ""

    def test_multiple_passages_separated(self):
        """Multiple passages should be separated by dividers."""
        chunks = [
            {"source_file": "a.pdf", "page_start": 1, "page_end": 1,
             "section_path": [], "text": "Text A"},
            {"source_file": "b.pdf", "page_start": 2, "page_end": 2,
             "section_path": [], "text": "Text B"},
        ]
        result = _build_context_text(chunks)
        assert "Text A" in result
        assert "Text B" in result
        assert "---" in result


class TestBuildImageText:
    """Tests for image text formatting."""

    def test_no_images_message(self):
        """Should return a 'no images' message when list is empty."""
        result = _build_image_text([])
        assert "No images" in result

    def test_formats_image_with_caption(self):
        """Should include caption and score."""
        images = [
            {"caption": "Wiring diagram", "image_type": "schematic",
             "page_number": 3, "composite_score": 0.85}
        ]
        result = _build_image_text(images)
        assert "Wiring diagram" in result
        assert "0.85" in result


class TestFallbackSynthesis:
    """Tests for the fallback synthesis (no LLM)."""

    def test_fallback_includes_query(self):
        """Fallback should reference the original query."""
        chunks = [
            {"source_file": "test.pdf", "page_start": 1,
             "page_end": 1, "text": "RAM installation steps."}
        ]
        result = _fallback_synthesis("How to install RAM?", chunks, [])
        assert "install RAM" in result

    def test_fallback_includes_source(self):
        """Fallback should cite the source document."""
        chunks = [
            {"source_file": "manual.pdf", "page_start": 5,
             "page_end": 5, "text": "Step 1: Open the clip."}
        ]
        result = _fallback_synthesis("test", chunks, [])
        assert "manual.pdf" in result


class TestExtractSources:
    """Tests for source citation extraction."""

    def test_unique_sources_only(self):
        """Should deduplicate identical source citations."""
        chunks = [
            {"source_file": "a.pdf", "page_start": 1, "page_end": 2,
             "section_path": []},
            {"source_file": "a.pdf", "page_start": 1, "page_end": 2,
             "section_path": []},
        ]
        sources = _extract_sources(chunks)
        assert len(sources) == 1

    def test_includes_section_path(self):
        """Source with section path should include it."""
        chunks = [
            {"source_file": "a.pdf", "page_start": 1, "page_end": 1,
             "section_path": ["Hardware", "RAM"]},
        ]
        sources = _extract_sources(chunks)
        assert "Hardware > RAM" in sources[0]


# ── Input Sanitization Tests ────────────────────────────────────────────

from app.core.sanitize import (
    sanitize_milvus_value,
    validate_identifier,
    build_milvus_filter,
)


class TestSanitizeMilvusValue:
    """Tests for Milvus expression sanitization."""

    def test_clean_value_passes_through(self):
        """Clean alphanumeric values should pass through unchanged."""
        assert sanitize_milvus_value("doc-123") == "doc-123"

    def test_strips_quotes(self):
        """Double quotes should be stripped to prevent breakout."""
        result = sanitize_milvus_value('value"injection')
        assert '"' not in result

    def test_strips_backslash(self):
        """Backslashes should be stripped."""
        result = sanitize_milvus_value("value\\escape")
        assert "\\" not in result

    def test_empty_value_raises(self):
        """Empty values should raise ValueError."""
        with pytest.raises(ValueError):
            sanitize_milvus_value("")

    def test_too_long_raises(self):
        """Values over 512 chars should raise ValueError."""
        with pytest.raises(ValueError):
            sanitize_milvus_value("x" * 600)


class TestBuildMilvusFilter:
    """Tests for the centralized Milvus filter builder."""

    def test_no_filters_returns_none(self):
        """No filters should return None."""
        assert build_milvus_filter() is None

    def test_doc_id_filter(self):
        """Should produce a doc_id filter expression."""
        expr = build_milvus_filter(doc_id="doc-001")
        assert 'doc_id == "doc-001"' in expr

    def test_combined_filter(self):
        """Should combine multiple filters with &&."""
        expr = build_milvus_filter(doc_id="doc-1", language="en")
        assert "doc_id" in expr
        assert "language" in expr
        assert "&&" in expr

    def test_injection_attempt_sanitized(self):
        """Injection attempts should be sanitized."""
        expr = build_milvus_filter(doc_id='x" || doc_id != "')
        assert '||' not in expr

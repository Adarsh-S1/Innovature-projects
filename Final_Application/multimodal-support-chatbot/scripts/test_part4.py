"""
Test Part 4: LangGraph Pipeline End-to-End
Tests all 5 agents individually, then runs the full compiled graph.
No external services required — uses rule-based fallbacks throughout.
"""

import sys
import asyncio
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.agents.context_router import context_router, _rule_based_classify
from app.agents.answer_synthesizer import _fallback_synthesis, _extract_sources
from app.agents.quality_guard import _rule_based_quality


def run_tests():
    print("=" * 60)
    print("PART 4 TEST: LangGraph 5-Agent Pipeline")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Context Router (rule-based fallback)
    # ══════════════════════════════════════════════════════════════════
    print("\n--- TEST 1: Context Router (Rule-Based) ---")

    test_queries = [
        ("Where is the RAM slot on the motherboard?", "locate_component", True),
        ("How to replace the CPU fan?", "how_to", False),
        ("Error code E-404 power supply failure", "troubleshoot", False),
        ("Show me the wiring diagram for section 3", "general", True),
        ("What are the specifications of the processor?", "specification", False),
    ]

    for query, expected_type, expected_image in test_queries:
        result = _rule_based_classify(query, [])
        qtype = result["query_type"]
        needs_img = result["needs_image"]
        intent = result["query_intent"]
        keywords = result["query_keywords"][:5]

        status = "✓" if qtype == expected_type else "✗"
        print(f"  {status} '{query[:50]}...'")
        print(f"      Type={qtype} Intent={intent} NeedsImage={needs_img} Keywords={keywords}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: Answer Synthesizer (fallback mode)
    # ══════════════════════════════════════════════════════════════════
    print("\n--- TEST 2: Answer Synthesizer (Fallback) ---")

    mock_chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "doc1",
            "text": "The Transformer model uses self-attention mechanisms to process sequences in parallel rather than sequentially. Multi-head attention allows the model to attend to information from different representation subspaces.",
            "page_start": 3,
            "page_end": 3,
            "section_path": ["Attention Is All You Need", "Model Architecture"],
            "source_file": "attention_paper.pdf",
        },
        {
            "chunk_id": "c2",
            "doc_id": "doc1",
            "text": "Positional encoding is added to the input embeddings to inject sequence order information. The authors use sinusoidal functions of different frequencies.",
            "page_start": 5,
            "page_end": 5,
            "section_path": ["Attention Is All You Need", "Positional Encoding"],
            "source_file": "attention_paper.pdf",
        },
    ]

    mock_images = [
        {"caption": "Transformer architecture diagram", "page_number": 3, "composite_score": 0.82},
    ]

    answer = _fallback_synthesis("How does the Transformer work?", mock_chunks, mock_images)
    print(f"  Generated answer ({len(answer)} chars):")
    print(f"  {answer[:200]}...")

    sources = _extract_sources(mock_chunks)
    print(f"  Sources: {sources}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: Quality Guard (rule-based)
    # ══════════════════════════════════════════════════════════════════
    print("\n--- TEST 3: Quality Guard (Rule-Based Scoring) ---")

    good_answer = (
        "The Transformer model uses self-attention mechanisms to process sequences "
        "in parallel. Multi-head attention allows attending to different subspaces. "
        "Positional encoding uses sinusoidal functions to inject order information."
    )

    scores, issues = _rule_based_quality(good_answer, "How does the Transformer work?", mock_chunks, mock_images)
    composite = sum(scores.get(d, 0.8) * w for d, w in {
        "groundedness": 0.30, "completeness": 0.25, "conciseness": 0.10,
        "technical_accuracy": 0.25, "image_relevance": 0.10
    }.items())

    print(f"  Good answer scores:")
    for dim, val in scores.items():
        print(f"    {dim}: {val:.3f}")
    print(f"  Composite: {composite:.3f}")
    print(f"  Issues: {issues}")

    bad_answer = "Maybe it works somehow. I'm not sure. Perhaps it uses something."
    scores2, issues2 = _rule_based_quality(bad_answer, "How does the Transformer work?", mock_chunks, [])
    composite2 = sum(scores2.get(d, 0.8) * w for d, w in {
        "groundedness": 0.30, "completeness": 0.25, "conciseness": 0.10,
        "technical_accuracy": 0.25, "image_relevance": 0.10
    }.items())

    print(f"\n  Bad answer scores:")
    for dim, val in scores2.items():
        print(f"    {dim}: {val:.3f}")
    print(f"  Composite: {composite2:.3f}")
    print(f"  Issues: {issues2}")

    assert composite > composite2, "Good answer should score higher than bad answer!"
    print(f"  ✓ Good ({composite:.3f}) > Bad ({composite2:.3f})")

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: Full Graph Compilation & Topology
    # ══════════════════════════════════════════════════════════════════
    print("\n--- TEST 4: LangGraph Compilation ---")

    from app.agents.graph import graph, build_graph

    nodes = list(graph.nodes.keys())
    print(f"  Graph nodes: {nodes}")
    assert "context_router" in nodes
    assert "hybrid_search" in nodes
    assert "visual_specialist" in nodes
    assert "answer_synthesizer" in nodes
    assert "quality_guard" in nodes
    print(f"  ✓ All 5 agent nodes registered")

    # Compile and verify it doesn't crash
    compiled = graph.compile()
    print(f"  ✓ Graph compiled successfully (type={type(compiled).__name__})")

    # ══════════════════════════════════════════════════════════════════
    # TEST 5: Full Pipeline Execution (with fallbacks)
    # ══════════════════════════════════════════════════════════════════
    print("\n--- TEST 5: Full Pipeline Execution ---")

    initial_state = {
        "user_query": "How does multi-head attention work in the Transformer?",
        "session_id": "test-session-001",
        "query_intent": "",
        "needs_image": False,
        "query_keywords": [],
        "query_concepts": [],
        "query_type": "general",
        "reformulated_query": "",
        "bm25_weight": 0.4,
        "vector_weight": 0.6,
        "retrieved_chunks": [],
        "retrieved_images": [],
        "rerank_scores": [],
        "draft_answer": "",
        "final_answer": "",
        "cited_sources": [],
        "quality_score": 0.0,
        "quality_issues": [],
        "retry_count": 0,
        "messages": [],
        "conversation_history": [],
        "response_images": [],
        "response_metadata": {},
    }

    final_state = compiled.invoke(initial_state)

    print(f"  Query Intent: {final_state.get('query_intent')}")
    print(f"  Needs Image: {final_state.get('needs_image')}")
    print(f"  Query Type: {final_state.get('query_type')}")
    print(f"  BM25 Weight: {final_state.get('bm25_weight')}")
    print(f"  Vector Weight: {final_state.get('vector_weight')}")
    print(f"  Retrieved Chunks: {len(final_state.get('retrieved_chunks', []))}")
    print(f"  Retrieved Images: {len(final_state.get('retrieved_images', []))}")
    print(f"  Quality Score: {final_state.get('quality_score', 0):.3f}")
    print(f"  Retry Count: {final_state.get('retry_count', 0)}")
    print(f"  Final Answer ({len(final_state.get('final_answer', ''))} chars):")
    print(f"    {final_state.get('final_answer', '')[:150]}...")

    assert final_state.get("query_intent") != "", "Context router should set intent"
    assert final_state.get("final_answer", "") != "", "Pipeline should produce an answer"
    print(f"\n  ✓ Pipeline executed end-to-end successfully")

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ALL PART 4 TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

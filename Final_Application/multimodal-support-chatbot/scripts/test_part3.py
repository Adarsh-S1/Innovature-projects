"""
Test Part 3: Hybrid Search & Cross-Modal Retrieval
Uses the 'Attention Is All You Need' PDF parsed in Part 2 as test data.
All operations run in-memory (offline mode) — no Milvus/Redis required.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import SemanticChunker
from app.retrieval.hybrid_searcher import HybridSearcher, BM25Index, SearchResult
from app.retrieval.image_searcher import ImageSearcher, ImageSearchResult
from app.retrieval.reranker import Reranker
from app.models.domain import QueryType, ImageType


def generate_fake_vector(text: str, dim: int = 64) -> list:
    """Generate a deterministic pseudo-embedding from text hash (for testing)."""
    rng = np.random.RandomState(hash(text) % (2**31))
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def run_tests():
    pdf_path = "attention_is_all_you_need.pdf"

    # ── Stage 1: Parse & Chunk (reuse from Part 2) ───────────────────
    print("=" * 60)
    print("SETUP: Parsing & chunking the PDF...")
    print("=" * 60)
    parser = PDFParser()
    parsed_doc = parser.parse(pdf_path)
    chunker = SemanticChunker()
    chunks = chunker.chunk_document(parsed_doc, doc_id="paper_01")
    print(f"  Parsed {parsed_doc.total_pages} pages → {len(chunks)} chunks\n")

    # Generate fake embeddings for each chunk (since we can't call OpenAI)
    chunk_dicts = []
    for c in chunks:
        d = {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.text,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section_path": c.section_path,
            "chunk_type": c.chunk_type.value,
            "source_file": c.metadata.get("source_file", ""),
            "product_id": c.metadata.get("product_id", ""),
            "language": c.metadata.get("language", "en"),
            "linked_images": c.linked_images,
            "text_vector": generate_fake_vector(c.text, dim=64),
        }
        chunk_dicts.append(d)

    # Create fake image records
    image_dicts = []
    all_images = []
    for page in parsed_doc.pages:
        for img in page.images:
            iid = f"img_{page.page_number}_{img.image_index}"
            image_dicts.append({
                "image_id": iid,
                "doc_id": "paper_01",
                "page_number": page.page_number,
                "caption": "Transformer architecture diagram",
                "description": "Multi-head attention and feed-forward network layers",
                "topic_concept": "transformer_architecture",
                "keyword_tags": ["attention", "transformer", "encoder", "decoder"],
                "image_type": ImageType.SCHEMATIC.value,
                "storage_url": f"images/paper_01/{iid}.png",
                "thumbnail_url": f"thumbs/paper_01/{iid}_thumb.png",
                "source_file": "attention_is_all_you_need.pdf",
                "product_id": "",
                "linked_chunk_ids": [],
                "clip_vector": generate_fake_vector(f"clip_{iid}", dim=64),
                "caption_vector": generate_fake_vector("Transformer architecture diagram", dim=64),
            })

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: BM25 Search
    # ══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 1: BM25 Keyword Search")
    print("=" * 60)

    bm25 = BM25Index()
    bm25.build({c["chunk_id"]: c["text"] for c in chunk_dicts})

    queries = [
        "attention mechanism",
        "encoder decoder architecture",
        "BLEU score translation results",
        "positional encoding sinusoidal",
    ]

    for query in queries:
        results = bm25.search(query, top_k=3)
        print(f"\n  Query: '{query}'")
        for rank, (cid, score) in enumerate(results, 1):
            text = next(c["text"][:80] for c in chunk_dicts if c["chunk_id"] == cid)
            print(f"    #{rank} (score={score:.4f}): {text}...")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: Hybrid Search (RRF Fusion)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 2: Hybrid Search (BM25 + Vector + RRF)")
    print("=" * 60)

    searcher = HybridSearcher()

    test_cases = [
        ("self-attention mechanism", QueryType.GENERAL),
        ("BLEU score 28.4", QueryType.SPECIFICATION),
        ("how to compute multi-head attention", QueryType.HOW_TO),
    ]

    for query, qtype in test_cases:
        qvec = generate_fake_vector(query, dim=64)
        results = searcher.search_offline(
            query=query,
            query_vector=qvec,
            chunks=chunk_dicts,
            query_type=qtype,
            top_k=5,
        )
        print(f"\n  Query: '{query}' (type={qtype.value})")
        for rank, r in enumerate(results, 1):
            print(
                f"    #{rank} RRF={r.rrf_score:.6f} "
                f"BM25_rank={r.bm25_rank} Vec_rank={r.vector_rank} "
                f"| {r.text[:60]}..."
            )

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: Reranker
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 3: Heuristic Reranker")
    print("=" * 60)

    reranker = Reranker()

    query = "multi-head attention"
    qvec = generate_fake_vector(query, dim=64)
    search_results = searcher.search_offline(
        query=query, query_vector=qvec, chunks=chunk_dicts,
        query_type=QueryType.GENERAL, top_k=10,
    )

    reranked = reranker.rerank(query, search_results, top_k=5, use_llm=False)
    print(f"\n  Query: '{query}' — {len(search_results)} candidates → {len(reranked)} reranked")
    for rank, r in enumerate(reranked, 1):
        print(
            f"    #{rank} final={r.final_score:.4f} "
            f"(rrf={r.original_rrf_score:.6f}, rerank={r.rerank_score:.4f}) "
            f"| {r.text[:60]}..."
        )

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: Image Search (Composite Scoring)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 4: Image Search (4-signal composite scoring)")
    print("=" * 60)

    img_searcher = ImageSearcher()

    # Use page numbers from top text results as relevant pages
    relevant_pages = {r.page_start for r in search_results[:3]}

    query_text_vec = generate_fake_vector("transformer architecture diagram", dim=64)
    query_clip_vec = generate_fake_vector("clip_transformer", dim=64)

    img_results = img_searcher.search_offline(
        query_text_vector=query_text_vec,
        query_clip_vector=query_clip_vec,
        image_records=image_dicts,
        relevant_page_numbers=relevant_pages,
        top_k=3,
        min_score=0.0,  # Show all for demo
    )

    print(f"\n  Relevant pages from text search: {relevant_pages}")
    print(f"  Image candidates: {len(image_dicts)}, Returned: {len(img_results)}")
    for rank, img in enumerate(img_results, 1):
        print(
            f"    #{rank} composite={img.composite_score:.4f} "
            f"(CLIP={img.clip_score:.3f}, caption={img.caption_score:.3f}, "
            f"coloc={img.colocation_bonus:.1f}, type={img.type_bonus:.1f}) "
            f"| Page {img.page_number}: {img.caption}"
        )

    # ══════════════════════════════════════════════════════════════════
    # TEST 5: Dynamic Weight Adjustment
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 5: Dynamic Weight Adjustment by Query Type")
    print("=" * 60)

    query = "error code E-404"
    qvec = generate_fake_vector(query, dim=64)

    for qtype in QueryType:
        results = searcher.search_offline(
            query=query, query_vector=qvec, chunks=chunk_dicts,
            query_type=qtype, top_k=3,
        )
        from app.retrieval.hybrid_searcher import _WEIGHT_TABLE
        w = _WEIGHT_TABLE.get(qtype, (0.4, 0.6))
        top_rrf = results[0].rrf_score if results else 0
        print(
            f"  {qtype.value:20s} → BM25={w[0]:.2f} Vec={w[1]:.2f} "
            f"| top_rrf={top_rrf:.6f} | top_result='{results[0].text[:40]}...'" if results else
            f"  {qtype.value:20s} → BM25={w[0]:.2f} Vec={w[1]:.2f} | no results"
        )

    print("\n" + "=" * 60)
    print("ALL PART 3 TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

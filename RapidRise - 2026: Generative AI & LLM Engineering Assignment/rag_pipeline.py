"""
rag_pipeline.py — RAG retrieval + generation with source citations.
"""

import json
import vector_db
import llm_client
import config


def retrieve(query: str, top_k: int = config.TOP_K_RESULTS) -> list[dict]:
    """Retrieve the top-K most relevant chunks for a query."""
    results = vector_db.search(query, top_k=top_k)
    print(f"[RAG] Retrieved {len(results)} chunks for query: '{query[:60]}...'")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r['source_document']} (p.{r['page_number']}) — similarity: {r['similarity']:.4f}")
    return results


def build_context(chunks: list[dict]) -> str:
    """Build a formatted context string from retrieved chunks with source labels."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk["source_document"]
        page = chunk["page_number"]
        context_parts.append(
            f"[Document: {source}, Page: {page}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_rag_answer(query: str, top_k: int = config.TOP_K_RESULTS) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, then generate an answer.

    Args:
        query: The user's question.
        top_k: Number of chunks to retrieve.

    Returns:
        dict with keys: summary, key_entities, confidence_score, sources, retrieved_chunks
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "chain_of_thought": "No relevant documents found in the knowledge base.",
            "summary": "I could not find relevant information in the document database to answer this query.",
            "key_entities": [],
            "confidence_score": 0.0,
            "sources": [],
            "retrieved_chunks": [],
        }

    # Step 2: Build context
    context = build_context(chunks)

    # Step 3: Generate with LLM
    result = llm_client.generate(query=query, context=context)

    # Step 4: Attach source metadata
    sources = list(set(c["source_document"] for c in chunks))
    result["sources"] = sources
    result["retrieved_chunks"] = [
        {"source": c["source_document"], "page": c["page_number"], "similarity": c["similarity"]}
        for c in chunks
    ]

    return result


if __name__ == "__main__":
    print("Testing RAG pipeline...")
    result = generate_rag_answer("What is the key innovation in the Transformer architecture?")
    print(json.dumps(result, indent=2))

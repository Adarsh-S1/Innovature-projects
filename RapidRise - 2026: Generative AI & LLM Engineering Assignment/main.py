"""
main.py — Main entry point for the Intelligent Research Assistant.

Usage:
    python main.py --setup          # Download PDFs, process, and ingest into vector DB
    python main.py --query "..."    # Ask a question (auto-routed)
    python main.py --demo           # Run routing demonstration
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import config


# ─── PDF Download URLs (arXiv) ────────────────────────────────────────────────
PDF_URLS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",
    "gpt2.pdf": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    "lora.pdf": "https://arxiv.org/pdf/2106.09685",
    "rag.pdf": "https://arxiv.org/pdf/2005.11401",
    "llama.pdf": "https://arxiv.org/pdf/2302.13971",
    "chain_of_thought.pdf": "https://arxiv.org/pdf/2201.11903",
}


def download_pdfs():
    """Download AI research papers to the documents directory."""
    docs_dir = config.DOCUMENTS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in PDF_URLS.items():
        filepath = docs_dir / filename
        if filepath.exists():
            print(f"  [✓] {filename} already exists, skipping.")
            continue

        print(f"  [↓] Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, str(filepath))
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"      Saved ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"      ERROR downloading {filename}: {e}")


def setup():
    """Full setup: download PDFs, process documents, ingest into vector DB."""
    print("=" * 70)
    print("  INTELLIGENT RESEARCH ASSISTANT — SETUP")
    print("=" * 70)

    # Step 1: Download PDFs
    print("\n[Step 1/4] Downloading PDF documents...")
    download_pdfs()

    # Step 2: Process documents
    print("\n[Step 2/4] Processing PDFs (extracting text and chunking)...")
    import document_processor
    chunks = document_processor.process_all_documents()

    if not chunks:
        print("ERROR: No chunks were created. Check the documents directory.")
        sys.exit(1)

    # Step 3: Setup database
    print("\n[Step 3/4] Setting up vector database...")
    import vector_db
    vector_db.setup_database()
    vector_db.clear_database()

    # Step 4: Ingest chunks
    print("\n[Step 4/4] Ingesting chunks into vector database...")
    vector_db.insert_chunks(chunks)

    count = vector_db.get_chunk_count()
    print(f"\n{'='*70}")
    print(f"  SETUP COMPLETE — {count} chunks in database")
    print(f"{'='*70}")


def query(question: str):
    """Route and answer a single question."""
    import router
    result = router.route_query(question)

    print(f"\n{'='*70}")
    print("  FINAL RESPONSE")
    print(f"{'='*70}")
    print(json.dumps(result, indent=2, default=str))
    return result


def demo():
    """Run the routing demonstration."""
    import router
    router.demo_routing()


def interactive():
    """Interactive REPL mode."""
    import router

    print("=" * 70)
    print("  INTELLIGENT RESEARCH ASSISTANT — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop. Type 'help' for commands.")
    print("=" * 70)

    while True:
        try:
            user_input = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "help":
            print("  Just type your question and press Enter.")
            print("  The system will auto-route to RAG or web search.")
            print("  Type 'quit' to exit.")
            continue

        result = router.route_query(user_input)
        print(f"\n📋 Route: {result.get('route_used', 'unknown').upper()}")
        print(f"📝 Summary: {result.get('summary', 'No response')}")
        print(f"🎯 Confidence: {result.get('confidence_score', 'N/A')}")

        entities = result.get("key_entities", [])
        if entities:
            print(f"🏷️  Entities: {', '.join(entities)}")

        sources = result.get("sources", [])
        if sources:
            if isinstance(sources[0], dict):
                print(f"🔗 Sources: {', '.join(s.get('title', s.get('url', '')) for s in sources)}")
            else:
                print(f"📄 Sources: {', '.join(sources)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Research Assistant")
    parser.add_argument("--setup", action="store_true", help="Download PDFs, process, and ingest into vector DB")
    parser.add_argument("--query", type=str, help="Ask a single question")
    parser.add_argument("--demo", action="store_true", help="Run routing demonstration with example queries")
    parser.add_argument("--interactive", action="store_true", help="Start interactive REPL mode")

    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.query:
        query(args.query)
    elif args.demo:
        demo()
    elif args.interactive:
        interactive()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. python main.py --setup        # First-time setup")
        print("  2. python main.py --demo          # Demo routing")
        print("  3. python main.py --interactive   # Interactive mode")

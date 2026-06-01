import sys
import os
import argparse
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ingestion.tasks import IngestionPipeline
from app.db.milvus_client import milvus_manager
from app.db.rustfs_client import rustfs_manager
from app.db.redis_client import redis_manager

async def ingest_directory(directory_path: str):
    # Initialize connections
    print("🔌 Initializing database connections...")
    await milvus_manager.connect()
    await milvus_manager.ensure_collections()
    await rustfs_manager.connect()
    await redis_manager.connect()
    print("✅ Databases connected!")
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        print(f"❌ Error: Directory '{directory_path}' does not exist.")
        return

    pdf_files = list(path.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in '{directory_path}'.")
        return

    print(f"Found {len(pdf_files)} PDF files in {directory_path}.")
    print("=" * 50)

    pipeline = IngestionPipeline()
    success_count = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Ingesting: {pdf_file.name}")
        try:
            # Generate a document ID
            doc_id = pdf_file.stem.replace(" ", "_").lower()
            
            # Remove any existing data for this document to prevent duplicates
            print(f"🧹 Removing old data for '{doc_id}' (if any)...")
            await milvus_manager.delete_document(doc_id)
            await rustfs_manager.delete_directory(f"images/{doc_id}/")
            await rustfs_manager.delete_directory(f"thumbs/{doc_id}/")
            
            # Read PDF as bytes
            with open(pdf_file, "rb") as f:
                pdf_bytes = f.read()

            # Run the synchronous pipeline
            result = pipeline.run(
                pdf_bytes=pdf_bytes,
                filename=pdf_file.name,
                doc_id=doc_id,
                product_id="general",
                language="en"
            )
            
            print(f"✅ Success! Chunks: {result.get('chunks_processed')}, Images: {result.get('images_processed')}")
            success_count += 1
        except Exception as e:
            if "rate limit" in str(e).lower() or "Too Many Requests" in str(e) or "LLMRateLimitError" in str(type(e)):
                print(f"\n⚠️  API Rate Limit Reached during '{pdf_file.name}'.")
                print("🛑 Stopping bulk ingestion safely. All previous documents have been saved.")
                break
            
            print(f"❌ Failed to ingest {pdf_file.name}: {e}")

    print("\n" + "=" * 50)
    print(f"Ingestion complete. Successfully ingested {success_count} out of {len(pdf_files)} files.")
    
    # Cleanup
    await redis_manager.disconnect()
    await milvus_manager.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest PDFs into the multimodal chatbot.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing PDF files")
    args = parser.parse_args()
    
    asyncio.run(ingest_directory(args.dir))

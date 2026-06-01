import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pymilvus import utility, connections
from app.core.config import get_settings

async def drop_all_collections():
    settings = get_settings()
    print("🔌 Connecting to Milvus...")
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )
    print("✅ Connected!")

    text_col = settings.MILVUS_TEXT_COLLECTION
    img_col = settings.MILVUS_IMAGE_COLLECTION

    if utility.has_collection(text_col):
        print(f"🗑️ Dropping collection: {text_col}")
        utility.drop_collection(text_col)
    else:
        print(f"⏭️ Collection {text_col} does not exist.")

    if utility.has_collection(img_col):
        print(f"🗑️ Dropping collection: {img_col}")
        utility.drop_collection(img_col)
    else:
        print(f"⏭️ Collection {img_col} does not exist.")

    # Drop MinIO bucket contents
    print("🔌 Connecting to MinIO...")
    from app.db.rustfs_client import rustfs_manager
    await rustfs_manager.connect()
    print(f"🗑️ Emptying MinIO bucket: {settings.RUSTFS_BUCKET_NAME}")
    await rustfs_manager.delete_directory("")

    print("\n✅ All collections and storage dropped successfully! The next time you run ingest_bulk.py, everything will be recreated fresh.")
    
    connections.disconnect("default")

if __name__ == "__main__":
    asyncio.run(drop_all_collections())

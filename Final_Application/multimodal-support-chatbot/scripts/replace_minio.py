import os
import glob

replacements = {
    "minio_client": "rustfs_client",
    "minio_manager": "rustfs_manager",
    "MinIOManager": "RustFSManager",
    "MinIOError": "RustFSError",
    "minio_connected": "rustfs_connected",
    "minio_connection_failed": "rustfs_connection_failed",
    "minio_bucket_created": "rustfs_bucket_created",
    "minio_file_uploaded": "rustfs_file_uploaded",
    "minio_file_deleted": "rustfs_file_deleted",
    "minio_directory_deleted": "rustfs_directory_deleted",
    "minio_directory_delete_failed": "rustfs_directory_delete_failed",
    "minio_ready": "rustfs_ready",
    "minio_startup_failed": "rustfs_startup_failed",
    "minio_upload_failed": "rustfs_upload_failed",
    "minioadmin": "rustfsadmin",
    "MINIO_": "RUSTFS_",
    "minio_ok": "rustfs_ok",
    "\"minio\"": "\"rustfs\"",
    "_upload_to_minio": "_upload_to_rustfs",
    "minio_data": "rustfs_data",
    "minio-server": "rustfs-server",
    "minio/minio:RELEASE.2024-09-22T00-33-43Z": "rustfs/rustfs:latest",
    'minio server /minio_data --console-address ":9001"': "rustfs /data",
    "/minio_data": "/data",
    "milvus-minio": "milvus-rustfs",
    "http://localhost:9000/minio/health/live": "http://localhost:9000/",
    "minio:9000": "rustfs:9000",
}

files_to_update = [
    "app/db/rustfs_client.py",
    "app/core/config.py",
    "app/core/exceptions.py",
    "app/main.py",
    "app/api/v1/health.py",
    "app/api/v1/chat.py",
    "app/ingestion/tasks.py",
    "scripts/drop_collections.py",
    "scripts/ingest_bulk.py",
    "tests/integration/test_api_endpoints.py",
    "infra/docker-compose.yml",
    ".env",
    "app/db/milvus_client.py" # just in case
]

for filepath in files_to_update:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()
        
        original_content = content
        for k, v in replacements.items():
            content = content.replace(k, v)
        
        # specific manual fix for docker-compose healthcheck
        if "infra/docker-compose.yml" in filepath:
            content = content.replace(
                'test: [ "CMD", "curl", "-f", "http://localhost:9000/" ]',
                'test: [ "CMD", "curl", "-f", "http://localhost:9000" ]'
            )
            # rename minio service block
            content = content.replace("  minio:", "  rustfs:")
            content = content.replace("      minio:", "      rustfs:")

        if content != original_content:
            with open(filepath, "w") as f:
                f.write(content)
            print(f"Updated {filepath}")
    else:
        print(f"File not found: {filepath}")

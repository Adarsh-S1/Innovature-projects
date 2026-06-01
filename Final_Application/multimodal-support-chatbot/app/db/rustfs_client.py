"""
MinIO (S3-compatible) client — object storage for PDFs and images.
"""

import io
from typing import Optional

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.core.exceptions import RustFSError
from app.core.logging import get_logger

logger = get_logger(__name__)


class RustFSManager:
    """
    Manages MinIO object storage for PDF files and extracted images.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = None

    async def connect(self) -> None:
        """Initialize the S3/MinIO boto3 client."""
        try:
            protocol = "https" if self._settings.RUSTFS_SECURE else "http"
            endpoint_url = f"{protocol}://{self._settings.RUSTFS_ENDPOINT}"

            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self._settings.RUSTFS_ACCESS_KEY,
                aws_secret_access_key=self._settings.RUSTFS_SECRET_KEY,
                config=BotoConfig(signature_version="s3v4"),
                region_name="us-east-1",
            )

            # Ensure the default bucket exists
            self._ensure_bucket(self._settings.RUSTFS_BUCKET_NAME)

            logger.info(
                "rustfs_connected",
                endpoint=self._settings.RUSTFS_ENDPOINT,
                bucket=self._settings.RUSTFS_BUCKET_NAME,
            )
        except Exception as e:
            logger.error("rustfs_connection_failed", error=str(e))
            raise RustFSError(f"Failed to connect to MinIO: {e}")

    def _ensure_bucket(self, bucket_name: str) -> None:
        """Create bucket if it does not exist."""
        try:
            self._client.head_bucket(Bucket=bucket_name)
        except ClientError:
            self._client.create_bucket(Bucket=bucket_name)
            logger.info("rustfs_bucket_created", bucket=bucket_name)

    @property
    def client(self):
        if not self._client:
            raise RustFSError("MinIO client not initialized. Call connect() first.")
        return self._client

    async def upload_file(
        self,
        file_data: bytes,
        object_key: str,
        content_type: str = "application/octet-stream",
        bucket: Optional[str] = None,
    ) -> str:
        """Upload a file to MinIO and return the object key."""
        bucket = bucket or self._settings.RUSTFS_BUCKET_NAME
        try:
            self.client.put_object(
                Bucket=bucket,
                Key=object_key,
                Body=io.BytesIO(file_data),
                ContentLength=len(file_data),
                ContentType=content_type,
            )
            logger.debug("rustfs_file_uploaded", key=object_key, bucket=bucket)
            return object_key
        except Exception as e:
            raise RustFSError(f"Failed to upload {object_key}: {e}")

    async def get_presigned_url(
        self,
        object_key: str,
        bucket: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Generate a presigned URL for an object."""
        bucket = bucket or self._settings.RUSTFS_BUCKET_NAME
        ttl = ttl or self._settings.RUSTFS_PRESIGNED_URL_TTL
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": object_key},
                ExpiresIn=ttl,
            )
            return url
        except Exception as e:
            raise RustFSError(f"Failed to generate presigned URL for {object_key}: {e}")

    async def download_file(
        self,
        object_key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """Download a file from MinIO."""
        bucket = bucket or self._settings.RUSTFS_BUCKET_NAME
        try:
            response = self.client.get_object(Bucket=bucket, Key=object_key)
            return response["Body"].read()
        except Exception as e:
            raise RustFSError(f"Failed to download {object_key}: {e}")

    async def delete_file(
        self,
        object_key: str,
        bucket: Optional[str] = None,
    ) -> None:
        """Delete a file from MinIO."""
        bucket = bucket or self._settings.RUSTFS_BUCKET_NAME
        try:
            self.client.delete_object(Bucket=bucket, Key=object_key)
            logger.debug("rustfs_file_deleted", key=object_key)
        except Exception as e:
            raise RustFSError(f"Failed to delete {object_key}: {e}")

    async def delete_directory(
        self,
        prefix: str,
        bucket: Optional[str] = None,
    ) -> None:
        """Delete all files under a specific prefix (folder)."""
        bucket = bucket or self._settings.RUSTFS_BUCKET_NAME
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        self.client.delete_object(Bucket=bucket, Key=obj['Key'])
            logger.debug("rustfs_directory_deleted", prefix=prefix)
        except Exception as e:
            logger.warning("rustfs_directory_delete_failed", prefix=prefix, error=str(e))

    async def health_check(self) -> bool:
        """Check if MinIO is reachable."""
        try:
            if self._client:
                self._client.list_buckets()
                return True
            return False
        except Exception:
            return False


# Singleton instance
rustfs_manager = RustFSManager()

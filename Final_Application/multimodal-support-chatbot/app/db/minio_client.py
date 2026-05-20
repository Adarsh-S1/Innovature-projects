"""
MinIO (S3-compatible) client — object storage for PDFs and images.
"""

import io
from typing import Optional

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.core.exceptions import MinIOError
from app.core.logging import get_logger

logger = get_logger(__name__)


class MinIOManager:
    """
    Manages MinIO object storage for PDF files and extracted images.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = None

    async def connect(self) -> None:
        """Initialize the S3/MinIO boto3 client."""
        try:
            protocol = "https" if self._settings.MINIO_SECURE else "http"
            endpoint_url = f"{protocol}://{self._settings.MINIO_ENDPOINT}"

            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self._settings.MINIO_ACCESS_KEY,
                aws_secret_access_key=self._settings.MINIO_SECRET_KEY,
                config=BotoConfig(signature_version="s3v4"),
                region_name="us-east-1",
            )

            # Ensure the default bucket exists
            self._ensure_bucket(self._settings.MINIO_BUCKET_NAME)

            logger.info(
                "minio_connected",
                endpoint=self._settings.MINIO_ENDPOINT,
                bucket=self._settings.MINIO_BUCKET_NAME,
            )
        except Exception as e:
            logger.error("minio_connection_failed", error=str(e))
            raise MinIOError(f"Failed to connect to MinIO: {e}")

    def _ensure_bucket(self, bucket_name: str) -> None:
        """Create bucket if it does not exist."""
        try:
            self._client.head_bucket(Bucket=bucket_name)
        except ClientError:
            self._client.create_bucket(Bucket=bucket_name)
            logger.info("minio_bucket_created", bucket=bucket_name)

    @property
    def client(self):
        if not self._client:
            raise MinIOError("MinIO client not initialized. Call connect() first.")
        return self._client

    async def upload_file(
        self,
        file_data: bytes,
        object_key: str,
        content_type: str = "application/octet-stream",
        bucket: Optional[str] = None,
    ) -> str:
        """Upload a file to MinIO and return the object key."""
        bucket = bucket or self._settings.MINIO_BUCKET_NAME
        try:
            self.client.put_object(
                Bucket=bucket,
                Key=object_key,
                Body=io.BytesIO(file_data),
                ContentLength=len(file_data),
                ContentType=content_type,
            )
            logger.debug("minio_file_uploaded", key=object_key, bucket=bucket)
            return object_key
        except Exception as e:
            raise MinIOError(f"Failed to upload {object_key}: {e}")

    async def get_presigned_url(
        self,
        object_key: str,
        bucket: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Generate a presigned URL for an object."""
        bucket = bucket or self._settings.MINIO_BUCKET_NAME
        ttl = ttl or self._settings.MINIO_PRESIGNED_URL_TTL
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": object_key},
                ExpiresIn=ttl,
            )
            return url
        except Exception as e:
            raise MinIOError(f"Failed to generate presigned URL for {object_key}: {e}")

    async def download_file(
        self,
        object_key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """Download a file from MinIO."""
        bucket = bucket or self._settings.MINIO_BUCKET_NAME
        try:
            response = self.client.get_object(Bucket=bucket, Key=object_key)
            return response["Body"].read()
        except Exception as e:
            raise MinIOError(f"Failed to download {object_key}: {e}")

    async def delete_file(
        self,
        object_key: str,
        bucket: Optional[str] = None,
    ) -> None:
        """Delete a file from MinIO."""
        bucket = bucket or self._settings.MINIO_BUCKET_NAME
        try:
            self.client.delete_object(Bucket=bucket, Key=object_key)
            logger.debug("minio_file_deleted", key=object_key)
        except Exception as e:
            raise MinIOError(f"Failed to delete {object_key}: {e}")

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
minio_manager = MinIOManager()

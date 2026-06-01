import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import io
import uuid
import sys

# ==========================================
# CONFIGURATION
# Update these values to match your Rustfs 
# endpoint and credentials.
# ==========================================
RUSTFS_ENDPOINT = "http://localhost:9000"  # e.g. where rustfs is running
ACCESS_KEY = "rustfsadmin"
SECRET_KEY = "rustfsadmin"
TEST_BUCKET = "rustfs-compatibility-test"

def run_compatibility_test():
    print(f"🔄 Connecting to Rustfs at {RUSTFS_ENDPOINT}...")
    
    # Milvus and the Python codebase strictly use boto3 with s3v4 signatures
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=RUSTFS_ENDPOINT,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return

    # TEST 1: Bucket Creation (Required by app and Milvus)
    print("\n[Test 1] Bucket Creation...")
    try:
        s3.head_bucket(Bucket=TEST_BUCKET)
        print(f"  ✅ Bucket '{TEST_BUCKET}' already exists.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            try:
                s3.create_bucket(Bucket=TEST_BUCKET)
                print(f"  ✅ Successfully created bucket '{TEST_BUCKET}'.")
            except Exception as e:
                print(f"  ❌ Failed to create bucket: {e}")
                return
        else:
            print(f"  ❌ Failed bucket head check: {e}")
            return

    # TEST 2: Put Object (Required by app for images, Milvus for segments)
    print("\n[Test 2] Put Object...")
    test_key = f"test-folder/test-image-{uuid.uuid4().hex[:6]}.txt"
    test_data = b"Hello, Rustfs! This is a test file to ensure object storage works."
    try:
        s3.put_object(
            Bucket=TEST_BUCKET,
            Key=test_key,
            Body=io.BytesIO(test_data),
            ContentLength=len(test_data),
            ContentType="text/plain",
        )
        print(f"  ✅ Successfully uploaded object to '{test_key}'.")
    except Exception as e:
        print(f"  ❌ Failed to put object: {e}")
        sys.exit(1)

    # TEST 3: Get Object (Required by app for downloading, Milvus for reading indexes)
    print("\n[Test 3] Get Object...")
    try:
        response = s3.get_object(Bucket=TEST_BUCKET, Key=test_key)
        downloaded_data = response['Body'].read()
        if downloaded_data == test_data:
            print("  ✅ Successfully downloaded and verified object content.")
        else:
            print("  ❌ Data mismatch! Object content was corrupted.")
    except Exception as e:
        print(f"  ❌ Failed to get object: {e}")

    # TEST 4: Generate Presigned URL (Required by app for frontend display)
    print("\n[Test 4] Generate Presigned URL...")
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": TEST_BUCKET, "Key": test_key},
            ExpiresIn=3600,
        )
        if url and url.startswith("http"):
            print("  ✅ Successfully generated presigned URL.")
            print(f"     URL: {url[:60]}...")
        else:
            print("  ❌ Generated URL looks invalid.")
    except Exception as e:
        print(f"  ❌ Failed to generate presigned URL: {e}")

    # TEST 5: List Objects Pagination (Required by app for directory deletion)
    print("\n[Test 5] List Objects (Pagination)...")
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=TEST_BUCKET, Prefix="test-folder/")
        found = False
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'] == test_key:
                        found = True
        
        if found:
            print("  ✅ Successfully listed objects and found the test file.")
        else:
            print("  ❌ Object not found in list response.")
    except Exception as e:
        print(f"  ❌ Failed to list objects: {e}")

    # TEST 6: Delete Object (Required by app and Milvus for compaction)
    print("\n[Test 6] Delete Object...")
    try:
        s3.delete_object(Bucket=TEST_BUCKET, Key=test_key)
        print("  ✅ Successfully deleted object.")
    except Exception as e:
        print(f"  ❌ Failed to delete object: {e}")

    print("\n" + "="*50)
    print("🎉 ALL TESTS COMPLETED. ")
    print("If all tests passed (✅), Rustfs implements the required S3 APIs")
    print("and is perfectly capable of replacing MinIO in this stack!")
    print("="*50)

if __name__ == "__main__":
    run_compatibility_test()

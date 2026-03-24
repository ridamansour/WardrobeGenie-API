"""
scripts/fetch_from_s3_or_api.py
===============================
Production ingestion script for WardrobeGenie.
Fetches user batch uploads from AWS S3 (or a centralized API) and safely stages
them into the local Airflow processing volume with idempotency and data validation.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 1. Production Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BatchIngestion")


def download_from_s3(batch_id: str, staging_dir: Path):
    """
    Production Logic: Downloads a batch of images from an AWS S3 ingestion bucket.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.critical("boto3 is required for S3 downloads but is not installed.")
        sys.exit(1)

    bucket_name = os.getenv("AWS_S3_INGESTION_BUCKET", "wardrobegenie-raw-uploads")
    prefix = f"batches/{batch_id}/"

    logger.info(f"Connecting to S3 Bucket: {bucket_name} | Prefix: {prefix}")

    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if 'Contents' not in response:
            logger.error(f"No objects found in S3 for batch: {batch_id}")
            sys.exit(1)  # Explicit failure for Airflow

        download_count = 0
        for obj in response['Contents']:
            file_key = obj['Key']
            if not file_key.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                logger.warning(f"Skipping non-image file: {file_key}")
                continue

            file_name = Path(file_key).name
            download_path = staging_dir / file_name

            # 2. Idempotency Check: Don't re-download if it exists and sizes match
            if download_path.exists() and download_path.stat().st_size == obj['Size']:
                logger.debug(f"Skipping existing file (Idempotent): {file_name}")
                continue

            logger.info(f"Downloading {file_name} from S3...")
            s3_client.download_file(bucket_name, file_key, str(download_path))
            download_count += 1

        logger.info(f"Successfully staged {download_count} files from S3.")

    except ClientError as e:
        logger.critical(f"AWS S3 Client Error: {e}")
        sys.exit(1)


def local_volume_validation(batch_id: str, staging_dir: Path):
    """
    Local Dev Logic: In our Docker Compose setup, the FastAPI server already
    saves uploads to the shared volume. We just validate their integrity.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    if not staging_dir.exists():
        logger.error(f"Staging directory missing: {staging_dir}. Did FastAPI save it correctly?")
        sys.exit(1)

    # 3. Data Integrity Check
    valid_files = [f for f in staging_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

    if not valid_files:
        logger.error(f"No valid images found in shared volume for batch {batch_id}.")
        sys.exit(1)

    logger.info(f"Local Validation: Verified {len(valid_files)} images ready for YOLOS processing.")


if __name__ == "__main__":
    # 4. Strict Command Line Argument Parsing
    parser = argparse.ArgumentParser(description="Ingest batch files for ML processing.")
    parser.add_argument('--batch_id', type=str, required=True, help="Unique identifier for the upload batch")
    args = parser.parse_args()

    target_dir = Path(f"/opt/airflow/data/user_uploads/{args.batch_id}")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Route logic based on Environment Variables (True CI/CD practice)
        if os.getenv("USE_S3_INGESTION", "false").lower() == "true":
            download_from_s3(args.batch_id, target_dir)
        else:
            local_volume_validation(args.batch_id, target_dir)

    except Exception as e:
        logger.critical(f"Fatal unhandled error during ingestion: {str(e)}")
        sys.exit(1)
"""R2 (S3-compatible) upload/download helpers."""
import boto3
from botocore.config import Config
from src import config

_client = None

def _s3():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=config.R2_ENDPOINT,
            aws_access_key_id=config.R2_ACCESS_KEY,
            aws_secret_access_key=config.R2_SECRET_KEY,
            config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "adaptive"}),
        )
    return _client


def upload(local_path: str, r2_key: str, content_type: str = "audio/mpeg") -> str:
    """Upload a local file to R2. Returns the public URL."""
    with open(local_path, "rb") as f:
        _s3().put_object(
            Bucket=config.R2_BUCKET,
            Key=r2_key,
            Body=f,
            ContentType=content_type,
        )
    return f"{config.R2_PUBLIC_URL}/{r2_key}"


def upload_bytes(data: bytes, r2_key: str, content_type: str = "audio/wav") -> str:
    _s3().put_object(
        Bucket=config.R2_BUCKET,
        Key=r2_key,
        Body=data,
        ContentType=content_type,
    )
    return f"{config.R2_PUBLIC_URL}/{r2_key}"


def download(r2_key: str, local_path: str):
    _s3().download_file(config.R2_BUCKET, r2_key, local_path)

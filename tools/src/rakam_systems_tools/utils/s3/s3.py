from __future__ import annotations

"""
Lightweight S3 utilities for rakam_systems.

This module provides a thin wrapper around boto3's S3 client with:

- Environment-based configuration (credentials, endpoint, region, bucket)
- Simple CRUD operations (upload, download, delete, list)
- Support for S3-compatible services (OVH, Scaleway, MinIO, etc.)
- Consistent error handling and logging
- A stable import path for all internal modules:

    >>> from rakam_systems_tools.utils import s3
    >>> s3.upload_file("my-key.txt", "Hello World")
    >>> content = s3.download_file("my-key.txt")  # Returns bytes
    >>> text = content.decode('utf-8')  # Decode to string if needed

Configuration is read from environment variables:
    - S3_ACCESS_KEY: AWS/S3 access key ID (required)
    - S3_SECRET_KEY: AWS/S3 secret access key (required)
    - S3_BUCKET_NAME: Default bucket name (required)
    - S3_ENDPOINT_URL: Custom endpoint for S3-compatible services (optional)
    - S3_REGION: AWS region or provider region code (default: "gra")
"""

import os
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    raise ImportError(
        "boto3 is required for S3 utilities. Install with: pip install boto3"
    )


# Configuration from environment variables
_S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
_S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
_S3_ENDPOINT_URL = os.getenv(
    "S3_ENDPOINT_URL", "https://s3.gra.io.cloud.ovh.net")
_S3_REGION = os.getenv("S3_REGION", "gra")
_S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Singleton client instance
_client: Optional[Any] = None


class S3Error(Exception):
    """Base exception for S3 operations."""
    pass


class S3ConfigError(S3Error):
    """Raised when S3 configuration is invalid or missing."""
    pass


class S3NotFoundError(S3Error):
    """Raised when a requested S3 object is not found."""
    pass


class S3PermissionError(S3Error):
    """Raised when access to S3 resource is denied."""
    pass


def _validate_config() -> None:
    """
    Validate that all required S3 configuration is present.

    Raises:
        S3ConfigError: If required configuration is missing.
    """
    missing = []
    if not _S3_ACCESS_KEY:
        missing.append("S3_ACCESS_KEY")
    if not _S3_SECRET_KEY:
        missing.append("S3_SECRET_KEY")
    if not _S3_BUCKET_NAME:
        missing.append("S3_BUCKET_NAME")

    if missing:
        raise S3ConfigError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Please set these in your environment or .env file."
        )


def get_client() -> Any:
    """
    Get or create the S3 client singleton.

    Returns:
        boto3.client: Configured S3 client instance.

    Raises:
        S3ConfigError: If required configuration is missing.
    """
    global _client

    if _client is not None:
        return _client

    _validate_config()

    client_config = {
        'aws_access_key_id': _S3_ACCESS_KEY,
        'aws_secret_access_key': _S3_SECRET_KEY,
        'region_name': _S3_REGION
    }

    # Add endpoint URL if specified (for S3-compatible services)
    if _S3_ENDPOINT_URL:
        client_config['endpoint_url'] = _S3_ENDPOINT_URL

    _client = boto3.client('s3', **client_config)
    return _client


def reset_client() -> None:
    """
    Reset the S3 client singleton.

    Useful for testing or when credentials need to be refreshed.
    """
    global _client
    _client = None


def upload_file(
    key: str,
    content: str | bytes,
    bucket: Optional[str] = None,
    content_type: str = "text/plain",
    metadata: Optional[Dict[str, str]] = None
) -> bool:
    """
    Upload a file to S3.

    Args:
        key: The S3 object key (path/filename).
        content: File content as string or bytes.
        bucket: Bucket name (defaults to S3_BUCKET_NAME).
        content_type: MIME type of the content.
        metadata: Optional metadata dictionary.

    Returns:
        bool: True if upload succeeded.

    Raises:
        S3Error: If upload fails.
    """
    client = get_client()
    bucket = bucket or _S3_BUCKET_NAME

    # Convert string to bytes if needed
    if isinstance(content, str):
        content = content.encode('utf-8')

    try:
        put_args = {
            'Bucket': bucket,
            'Key': key,
            'Body': content,
            'ContentType': content_type
        }

        if metadata:
            put_args['Metadata'] = metadata

        client.put_object(**put_args)
        return True

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        raise S3Error(f"Failed to upload file '{key}': {error_code} - {e}")


def download_file(
    key: str,
    bucket: Optional[str] = None
) -> bytes:
    """
    Download a file from S3.

    Args:
        key: The S3 object key (path/filename).
        bucket: Bucket name (defaults to S3_BUCKET_NAME).

    Returns:
        bytes: File content as bytes. The application should decode if needed
               (e.g., content.decode('utf-8') for text files).

    Raises:
        S3NotFoundError: If file does not exist.
        S3Error: If download fails.

    Note:
        This function always returns bytes to preserve data integrity and support
        all file types (text, binary, images, PDFs, etc.). For text files, decode
        the result: content.decode('utf-8')
    """
    client = get_client()
    bucket = bucket or _S3_BUCKET_NAME

    try:
        response = client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()
        return content

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey' or error_code == '404':
            raise S3NotFoundError(f"File not found: '{key}'")
        raise S3Error(f"Failed to download file '{key}': {error_code} - {e}")


def delete_file(key: str, bucket: Optional[str] = None) -> bool:
    """
    Delete a file from S3.

    Args:
        key: The S3 object key (path/filename).
        bucket: Bucket name (defaults to S3_BUCKET_NAME).

    Returns:
        bool: True if deletion succeeded.

    Raises:
        S3Error: If deletion fails.
    """
    client = get_client()
    bucket = bucket or _S3_BUCKET_NAME

    try:
        client.delete_object(Bucket=bucket, Key=key)
        return True

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        raise S3Error(f"Failed to delete file '{key}': {error_code} - {e}")


def file_exists(key: str, bucket: Optional[str] = None) -> bool:
    """
    Check if a file exists in S3.

    Args:
        key: The S3 object key (path/filename).
        bucket: Bucket name (defaults to S3_BUCKET_NAME).

    Returns:
        bool: True if file exists, False otherwise.
    """
    client = get_client()
    bucket = bucket or _S3_BUCKET_NAME

    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey' or error_code == '404':
            return False
        # Re-raise other errors
        raise S3Error(
            f"Failed to check file existence '{key}': {error_code} - {e}")


def list_files(
    prefix: str = "",
    bucket: Optional[str] = None,
    max_keys: int = 1000
) -> List[Dict[str, Any]]:
    """
    List files in S3 bucket with optional prefix filter.

    Args:
        prefix: Filter results to keys starting with this prefix.
        bucket: Bucket name (defaults to S3_BUCKET_NAME).
        max_keys: Maximum number of keys to return.

    Returns:
        List[Dict]: List of file metadata dictionaries with keys:
            - Key: Object key (str)
            - Size: File size in bytes (int)
            - LastModified: Last modification datetime
            - ETag: Entity tag (str)
    """
    client = get_client()
    bucket = bucket or _S3_BUCKET_NAME

    try:
        response = client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=max_keys
        )

        if 'Contents' not in response:
            return []

        return [
            {
                'Key': obj['Key'],
                'Size': obj['Size'],
                'LastModified': obj['LastModified'],
                'ETag': obj.get('ETag', '')
            }
            for obj in response['Contents']
        ]

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        raise S3Error(
            f"Failed to list files with prefix '{prefix}': {error_code} - {e}")


def get_file_metadata(key: str, bucket: Optional[str] = None) -> Dict[str, Any]:
    """
    Get metadata for a file without downloading it.

    Args:
        key: The S3 object key (path/filename).
        bucket: Bucket name (defaults to S3_BUCKET_NAME).

    Returns:
        Dict: Metadata dictionary with keys like ContentType, ContentLength, 
              LastModified, ETag, Metadata (custom metadata).

    Raises:
        S3NotFoundError: If file does not exist.
        S3Error: If operation fails.
    """
    client = get_client()
    bucket = bucket or _S3_BUCKET_NAME

    try:
        response = client.head_object(Bucket=bucket, Key=key)
        return {
            'ContentType': response.get('ContentType', ''),
            'ContentLength': response.get('ContentLength', 0),
            'LastModified': response.get('LastModified'),
            'ETag': response.get('ETag', ''),
            'Metadata': response.get('Metadata', {})
        }

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey' or error_code == '404':
            raise S3NotFoundError(f"File not found: '{key}'")
        raise S3Error(
            f"Failed to get metadata for '{key}': {error_code} - {e}")


def create_bucket(bucket: Optional[str] = None) -> bool:
    """
    Create an S3 bucket.

    Args:
        bucket: Bucket name (defaults to S3_BUCKET_NAME).

    Returns:
        bool: True if bucket was created or already exists.

    Raises:
        S3Error: If bucket creation fails.
    """
    client = get_client()
    bucket = bucket or _S3_BUCKET_NAME

    try:
        client.create_bucket(Bucket=bucket)
        return True

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')

        # Bucket already exists cases
        if error_code in ('BucketAlreadyOwnedByYou', 'BucketAlreadyExists'):
            return True

        raise S3Error(
            f"Failed to create bucket '{bucket}': {error_code} - {e}")


def bucket_exists(bucket: Optional[str] = None) -> bool:
    """
    Check if a bucket exists and is accessible.

    Args:
        bucket: Bucket name (defaults to S3_BUCKET_NAME).

    Returns:
        bool: True if bucket exists and is accessible.
    """
    client = get_client()
    bucket = bucket or _S3_BUCKET_NAME

    try:
        client.head_bucket(Bucket=bucket)
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code in ('404', 'NoSuchBucket'):
            return False
        if error_code == '403':
            raise S3PermissionError(f"Access denied to bucket '{bucket}'")
        raise S3Error(f"Failed to check bucket '{bucket}': {error_code} - {e}")


def list_buckets() -> List[Dict[str, Any]]:
    """
    List all buckets in the account.

    Returns:
        List[Dict]: List of bucket metadata dictionaries with keys:
            - Name: Bucket name (str)
            - CreationDate: Creation datetime
    """
    client = get_client()

    try:
        response = client.list_buckets()
        return [
            {
                'Name': bucket['Name'],
                'CreationDate': bucket['CreationDate']
            }
            for bucket in response.get('Buckets', [])
        ]

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        raise S3Error(f"Failed to list buckets: {error_code} - {e}")


def get_config() -> Dict[str, Any]:
    """
    Get current S3 configuration.

    Returns:
        Dict: Configuration dictionary (secrets are masked).
    """
    return {
        'bucket': _S3_BUCKET_NAME,
        'region': _S3_REGION,
        'endpoint_url': _S3_ENDPOINT_URL,
        'access_key': _S3_ACCESS_KEY[:4] + '***' if _S3_ACCESS_KEY else None,
        'has_secret_key': bool(_S3_SECRET_KEY)
    }


__all__ = [
    # Exceptions
    "S3Error",
    "S3ConfigError",
    "S3NotFoundError",
    "S3PermissionError",
    # Client management
    "get_client",
    "reset_client",
    "get_config",
    # File operations
    "upload_file",
    "download_file",
    "delete_file",
    "file_exists",
    "list_files",
    "get_file_metadata",
    # Bucket operations
    "create_bucket",
    "bucket_exists",
    "list_buckets",
]

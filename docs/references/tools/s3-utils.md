---
title: S3 Component
description: Lightweight Pythonic wrapper for S3 operations in rakam-system-tools

# S3 Component

The S3 component provides a lightweight, Pythonic wrapper around boto3 for S3 operations, following the Rakam Systems (RS) component pattern.

## Features

- **Simple API**: Clean, intuitive functions for common S3 operations
- **Environment-based Configuration**: No hardcoded credentials
- **S3-Compatible Services**: Works with AWS S3, OVH, Scaleway, MinIO, etc.
- **Proper Error Handling**: Custom exceptions for different error cases
- **Singleton Client**: Efficient connection reuse
- **Type Hints**: Full type annotations for better IDE support
- **RS Component Style**: Consistent with other Rakam Systems utilities

## Installation

Install **rakam-system-tools** (rs_core). The S3 component is included and pulls in boto3 as a dependency:

```bash
pip install rakam-system-tools
```

Or from the repo (development/editable):

```bash
# From the repository root
pip install  rakam-system-tools
# or with uv
uv pip install  rakam-system-tools
```

Or add to your `requirements.txt`:

```
rakam-system-tools
```

## Configuration

Set these environment variables (e.g., in your `.env` file):

```bash
# Required
S3_ACCESS_KEY=your_access_key_here
S3_SECRET_KEY=your_secret_key_here
S3_BUCKET_NAME=your-bucket-name

# Optional (for S3-compatible services like OVH, Scaleway, MinIO)
S3_ENDPOINT_URL=https://s3.gra.io.cloud.ovh.net
S3_REGION=gra
```

### Provider-Specific Endpoints

**OVH Object Storage:**

```bash
S3_ENDPOINT_URL=https://s3.gra.io.cloud.ovh.net  # Gravelines
S3_REGION=gra
```

**Scaleway Object Storage:**

```bash
S3_ENDPOINT_URL=https://s3.fr-par.scw.cloud
S3_REGION=fr-par
```

**MinIO (self-hosted):**

```bash
S3_ENDPOINT_URL=http://localhost:9000
S3_REGION=us-east-1
```

**AWS S3 (default):**

```bash
# No endpoint URL needed for AWS
S3_REGION=us-east-1
```

## Quick Start

After installing `rakam-system-tools` and setting the required env vars (see [Configuration](#configuration)):

```python
from rakam_systems_tools.utils import s3

# Upload a file
s3.upload_file(
    key="documents/report.txt",
    content="Hello World!",
    content_type="text/plain"
)

# Download a file (returns bytes; decode for text)
content = s3.download_file("documents/report.txt")
print(content.decode("utf-8"))  # "Hello World!"

# Check if file exists
if s3.file_exists("documents/report.txt"):
    print("File exists!")

# List files with prefix
files = s3.list_files(prefix="documents/")
for file in files:
    print(f"{file['Key']} - {file['Size']} bytes")

# Delete a file
s3.delete_file("documents/report.txt")
```

## Basic Usage Example

Minimal script you can run locally (set `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_NAME` in `.env` or environment):

```python
# quick_s3_example.py
from rakam_systems_tools.utils import s3
from rakam_systems_tools.utils.s3 import S3Error, S3ConfigError

try:
    # Ensure bucket is reachable
    if not s3.bucket_exists():
        raise SystemExit("Bucket not found or not accessible. Check env vars.")

    # Upload
    s3.upload_file("demo/hello.txt", "Hello from rs_core!", content_type="text/plain")
    print("Uploaded demo/hello.txt")

    # Download
    data = s3.download_file("demo/hello.txt")
    print("Downloaded:", data.decode("utf-8"))

    # List
    for f in s3.list_files(prefix="demo/"):
        print(f"  {f['Key']} ({f['Size']} bytes)")

    # Delete
    s3.delete_file("demo/hello.txt")
    print("Deleted demo/hello.txt")
except S3ConfigError as e:
    print("Config error:", e)
except S3Error as e:
    print("S3 error:", e)
```

## Full Example (Run from Repo)

A full runnable example that demonstrates upload, download, list, metadata, bucket ops, and error handling lives in the repo:

```bash
# From repository root; ensure .env has S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME
python examples/s3_examples/s3_operations_example.py
```

Clean up test objects created by that example:

```bash
python examples/s3_examples/s3_operations_example.py --cleanup
```

## API Reference

### File Operations

#### `upload_file(key, content, bucket=None, content_type="text/plain", metadata=None)`

Upload a file to S3.

**Parameters:**

- `key` (str): The S3 object key (path/filename)
- `content` (str | bytes): File content
- `bucket` (str, optional): Bucket name (defaults to `S3_BUCKET_NAME`)
- `content_type` (str): MIME type (default: "text/plain")
- `metadata` (dict, optional): Custom metadata dictionary

**Returns:** `bool` - True if successful

**Raises:** `S3Error` if upload fails

**Example:**

```python
s3.upload_file(
    key="data/config.json",
    content='{"setting": "value"}',
    content_type="application/json",
    metadata={"version": "1.0"}
)
```


#### `download_file(key, bucket=None, as_bytes=False)`

Download a file from S3.

**Parameters:**

- `key` (str): The S3 object key
- `bucket` (str, optional): Bucket name
- `as_bytes` (bool): Return bytes instead of string (default: False)

**Returns:** `str | bytes` - File content

**Raises:**

- `S3NotFoundError` if file doesn't exist
- `S3Error` if download fails

**Example:**

```python
# Download as string (decode bytes for text)
text = s3.download_file("data/config.json").decode("utf-8")

# Download as bytes (for binary files)
image_data = s3.download_file("images/photo.jpg", as_bytes=True)
```


#### `delete_file(key, bucket=None)`

Delete a file from S3.

**Parameters:**

- `key` (str): The S3 object key
- `bucket` (str, optional): Bucket name

**Returns:** `bool` - True if successful

**Raises:** `S3Error` if deletion fails


#### `file_exists(key, bucket=None)`

Check if a file exists in S3.

**Parameters:**

- `key` (str): The S3 object key
- `bucket` (str, optional): Bucket name

**Returns:** `bool` - True if file exists

**Raises:** `S3Error` on errors other than "not found"


#### `list_files(prefix="", bucket=None, max_keys=1000)`

List files in S3 bucket with optional prefix filter.

**Parameters:**

- `prefix` (str): Filter to keys starting with this prefix
- `bucket` (str, optional): Bucket name
- `max_keys` (int): Maximum number of keys to return (default: 1000)

**Returns:** `List[Dict]` - List of file metadata dictionaries

Each dictionary contains:

- `Key` (str): Object key
- `Size` (int): File size in bytes
- `LastModified` (datetime): Last modification time
- `ETag` (str): Entity tag

**Example:**

```python
# List all files in a folder
files = s3.list_files(prefix="documents/2024/")
for file in files:
    print(f"{file['Key']}: {file['Size']} bytes")
```


#### `get_file_metadata(key, bucket=None)`

Get metadata for a file without downloading it.

**Parameters:**

- `key` (str): The S3 object key
- `bucket` (str, optional): Bucket name

**Returns:** `Dict` - Metadata dictionary

Contains:

- `ContentType` (str): MIME type
- `ContentLength` (int): File size in bytes
- `LastModified` (datetime): Last modification time
- `ETag` (str): Entity tag
- `Metadata` (dict): Custom metadata

**Raises:**

- `S3NotFoundError` if file doesn't exist
- `S3Error` on other failures


### Bucket Operations

#### `create_bucket(bucket=None)`

Create an S3 bucket.

**Parameters:**

- `bucket` (str, optional): Bucket name (defaults to `S3_BUCKET_NAME`)

**Returns:** `bool` - True if created or already exists

**Raises:** `S3Error` if creation fails


#### `bucket_exists(bucket=None)`

Check if a bucket exists and is accessible.

**Parameters:**

- `bucket` (str, optional): Bucket name

**Returns:** `bool` - True if bucket exists

**Raises:** `S3PermissionError` if access is denied


#### `list_buckets()`

List all buckets in the account.

**Returns:** `List[Dict]` - List of bucket metadata

Each dictionary contains:

- `Name` (str): Bucket name
- `CreationDate` (datetime): Creation time


### Client Management

#### `get_client()`

Get or create the S3 client singleton.

**Returns:** `boto3.client` - Configured S3 client

**Raises:** `S3ConfigError` if configuration is missing


#### `reset_client()`

Reset the S3 client singleton. Useful for testing or credential refresh.


#### `get_config()`

Get current S3 configuration (with masked secrets).

**Returns:** `Dict` - Configuration dictionary


## Exception Handling

The component provides custom exceptions for better error handling:

```python
from rakam_systems_tools.utils import s3

try:
    content = s3.download_file("missing-file.txt")
except s3.S3NotFoundError:
    print("File not found")
except s3.S3PermissionError:
    print("Access denied")
except s3.S3ConfigError:
    print("Configuration error")
except s3.S3Error as e:
    print(f"General S3 error: {e}")
```

### Exception Hierarchy

- `S3Error` - Base exception for all S3 operations
  - `S3ConfigError` - Configuration is invalid or missing
  - `S3NotFoundError` - Requested object not found
  - `S3PermissionError` - Access denied

## Advanced Examples

### Working with Binary Files

```python
# Upload binary file
with open("image.jpg", "rb") as f:
    image_data = f.read()

s3.upload_file(
    key="images/photo.jpg",
    content=image_data,
    content_type="image/jpeg"
)

# Download binary file
image_data = s3.download_file("images/photo.jpg", as_bytes=True)
with open("downloaded.jpg", "wb") as f:
    f.write(image_data)
```

### Using Custom Metadata

```python
from datetime import datetime

# Upload with metadata
s3.upload_file(
    key="reports/monthly.pdf",
    content=pdf_data,
    content_type="application/pdf",
    metadata={
        "author": "John Doe",
        "created": datetime.now().isoformat(),
        "version": "1.0"
    }
)

# Retrieve metadata
metadata = s3.get_file_metadata("reports/monthly.pdf")
print(f"Author: {metadata['Metadata']['author']}")
```

### Batch Operations

```python
# Upload multiple files
files_to_upload = [
    ("data/file1.txt", "Content 1"),
    ("data/file2.txt", "Content 2"),
    ("data/file3.txt", "Content 3"),
]

for key, content in files_to_upload:
    try:
        s3.upload_file(key, content)
        print(f"✅ Uploaded {key}")
    except s3.S3Error as e:
        print(f"❌ Failed to upload {key}: {e}")

# Delete multiple files
files = s3.list_files(prefix="data/")
for file in files:
    s3.delete_file(file['Key'])
```

### Working with Different Buckets

```python
# Upload to specific bucket
s3.upload_file(
    key="file.txt",
    content="Hello",
    bucket="my-other-bucket"
)

# List files in specific bucket
files = s3.list_files(prefix="", bucket="my-other-bucket")
```

## Testing

Run the full S3 operations example (from the repository root):

```bash
python examples/s3_examples/s3_operations_example.py
```

Clean up test files created by the example:

```bash
python examples/s3_examples/s3_operations_example.py --cleanup
```

## Comparison with Original Script

### Original Script (boto3 direct usage)

```python
import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    endpoint_url=S3_ENDPOINT_URL,
    region_name=S3_REGION
)

try:
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=file_key,
        Body=content.encode('utf-8'),
        ContentType='text/plain'
    )
except ClientError as e:
    print(f"Error: {e}")
```

### RS Component Style (clean and simple)

```python
from rakam_systems_tools.utils import s3

try:
    s3.upload_file(file_key, content)
except s3.S3Error as e:
    print(f"Error: {e}")
```

## Best Practices

1. **Use environment variables** for configuration (never hardcode credentials)
2. **Handle exceptions** appropriately for your use case
3. **Use prefixes** to organize files in folders (e.g., "documents/2024/")
4. **Set appropriate content types** for better browser handling
5. **Add metadata** for tracking and auditing
6. **Clean up test files** after development/testing
7. **Use `as_bytes=True`** when working with binary files

## Troubleshooting

### Configuration Errors

```
S3ConfigError: Missing required environment variables: S3_ACCESS_KEY, S3_SECRET_KEY
```

**Solution:** Ensure all required environment variables are set in your `.env` file.

### Connection Errors

```
S3Error: Failed to upload file 'test.txt': EndpointConnectionError
```

**Solution:** Check that `S3_ENDPOINT_URL` is correct for your S3-compatible service.

### Permission Errors

```
S3PermissionError: Access denied to bucket 'my-bucket'
```

**Solution:** Verify your credentials have the necessary permissions for the bucket.

### Bucket Name Errors

```
S3Error: Invalid bucket name
```

**Solution:** Bucket names must:

- Be 3-63 characters long
- Contain only lowercase letters, numbers, and hyphens
- Start and end with a letter or number
- NOT contain underscores or uppercase letters


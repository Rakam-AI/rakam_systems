---
title: Cloud Storage (S3)
---

# Cloud Storage (S3)

The `rakam-system-tools` package includes a lightweight wrapper around boto3 for S3-compatible storage (AWS S3, OVH, Scaleway, MinIO).

```bash
pip install rakam-system-tools
```

## Configure S3

Set these environment variables in your `.env` file:

```bash
# Required
S3_ACCESS_KEY=your_access_key_here
S3_SECRET_KEY=your_secret_key_here
S3_BUCKET_NAME=your-bucket-name

# Optional (for S3-compatible services like OVH, Scaleway, MinIO)
S3_ENDPOINT_URL=https://s3.gra.io.cloud.ovh.net
S3_REGION=gra
```

For AWS S3, omit `S3_ENDPOINT_URL`. For other providers, set it to their endpoint (e.g., `https://s3.fr-par.scw.cloud` for Scaleway, `http://localhost:9000` for MinIO).

## Use S3

```python
from rakam_systems_tools.utils import s3

# Upload
s3.upload_file(
    key="documents/report.txt",
    content="Hello World!",
    content_type="text/plain"
)

# Download (returns bytes; decode for text)
content = s3.download_file("documents/report.txt")
print(content.decode("utf-8"))

# Check existence
if s3.file_exists("documents/report.txt"):
    print("File exists!")

# List files with prefix
files = s3.list_files(prefix="documents/")
for file in files:
    print(f"{file['Key']} - {file['Size']} bytes")

# Delete
s3.delete_file("documents/report.txt")
```

## Handle S3 errors

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

Exception hierarchy: `S3Error` → `S3ConfigError`, `S3NotFoundError`, `S3PermissionError`.

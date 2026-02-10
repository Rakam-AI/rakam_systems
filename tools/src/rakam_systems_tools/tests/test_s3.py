"""
Tests for rakam_system_core S3 utilities.

Requires S3_ACCESS_KEY, S3_SECRET_KEY, and S3_BUCKET_NAME in the environment
(or .env). All tests are skipped if config is missing. Test keys use the
prefix pytest_s3_test/ and are deleted after the run.
"""
import os
import pytest
from datetime import datetime

# Only import after we know we might run S3 tests
try:
    from rakam_systems_tools.utils.s3 import s3
    from rakam_systems_tools.utils.s3.s3 import (
        S3Error,
        S3ConfigError,
        S3NotFoundError,
        S3PermissionError,
    )
    HAS_S3 = True
except ImportError:
    HAS_S3 = False

TEST_PREFIX = "pytest_s3_test/"


def _s3_config_ok() -> bool:
    """Return True if S3 env vars are set so we can run live tests."""
    return bool(
        os.getenv("S3_ACCESS_KEY")
        and os.getenv("S3_SECRET_KEY")
        and os.getenv("S3_BUCKET_NAME")
    )


@pytest.fixture(scope="module")
def s3_test_key():
    """A unique key for this test run to avoid collisions."""
    return f"{TEST_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}_test.txt"


@pytest.fixture(scope="module")
def s3_require_config():
    """Skip the whole module if S3 config is not available."""
    if not HAS_S3:
        pytest.skip("rakam_system_core.ai_utils.s3 not available")
    if not _s3_config_ok():
        pytest.skip(
            "S3 tests require S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME"
        )
    yield
    # Teardown: delete test objects created under TEST_PREFIX
    try:
        for f in s3.list_files(prefix=TEST_PREFIX):
            s3.delete_file(f["Key"])
    except (S3Error, S3ConfigError):
        pass


# ---- Config and client ----

def test_get_config(s3_require_config):
    """get_config returns a dict with masked secrets."""
    cfg = s3.get_config()
    assert isinstance(cfg, dict)
    assert "bucket" in cfg
    assert cfg["bucket"] == os.getenv("S3_BUCKET_NAME")
    assert "access_key" in cfg
    assert "***" in str(cfg.get("access_key", ""))


# ---- Bucket ----

def test_bucket_exists(s3_require_config):
    """bucket_exists returns True for configured bucket."""
    assert s3.bucket_exists() is True


def test_list_buckets(s3_require_config):
    """list_buckets returns a list; default bucket should be in it."""
    buckets = s3.list_buckets()
    assert isinstance(buckets, list)
    names = [b["Name"] for b in buckets]
    assert os.getenv("S3_BUCKET_NAME") in names


# ---- File CRUD ----

def test_upload_download_roundtrip(s3_require_config, s3_test_key):
    """Upload then download returns same content."""
    content = "Hello S3 test at " + datetime.now().isoformat()
    s3.upload_file(key=s3_test_key, content=content, content_type="text/plain")
    data = s3.download_file(key=s3_test_key)
    assert isinstance(data, bytes)
    assert data.decode("utf-8") == content


def test_file_exists(s3_require_config, s3_test_key):
    """file_exists is True after upload, False for missing key."""
    s3.upload_file(key=s3_test_key, content="x", content_type="text/plain")
    assert s3.file_exists(s3_test_key) is True
    assert s3.file_exists(TEST_PREFIX + "nonexistent_key_12345.txt") is False


def test_get_file_metadata(s3_require_config, s3_test_key):
    """get_file_metadata returns ContentType, ContentLength, etc."""
    s3.upload_file(
        key=s3_test_key,
        content="meta test",
        content_type="text/plain",
        metadata={"custom": "value"},
    )
    meta = s3.get_file_metadata(s3_test_key)
    assert meta["ContentType"] == "text/plain"
    assert meta["ContentLength"] == len(b"meta test")
    assert "LastModified" in meta
    assert meta.get("Metadata", {}).get("custom") == "value"


def test_download_not_found(s3_require_config):
    """download_file raises S3NotFoundError for missing key."""
    with pytest.raises(S3NotFoundError):
        s3.download_file(TEST_PREFIX + "does_not_exist_xyz.txt")


def test_list_files(s3_require_config, s3_test_key):
    """list_files with prefix returns entries with Key, Size, LastModified."""
    s3.upload_file(key=s3_test_key, content="list test",
                   content_type="text/plain")
    files = s3.list_files(prefix=TEST_PREFIX)
    assert isinstance(files, list)
    keys = [f["Key"] for f in files]
    assert s3_test_key in keys
    for f in files:
        assert "Key" in f and "Size" in f and "LastModified" in f


def test_delete_file(s3_require_config, s3_test_key):
    """delete_file removes the object; file_exists is False after."""
    s3.upload_file(key=s3_test_key, content="to delete",
                   content_type="text/plain")
    assert s3.file_exists(s3_test_key) is True
    s3.delete_file(s3_test_key)
    assert s3.file_exists(s3_test_key) is False


def test_upload_binary(s3_require_config, s3_test_key):
    """Upload bytes (binary) and download unchanged."""
    key_bin = s3_test_key.replace(".txt", "_bin.bin")
    data = b"\x00\x01\x02\xff\xfe"
    s3.upload_file(key=key_bin, content=data,
                   content_type="application/octet-stream")
    out = s3.download_file(key=key_bin)
    assert out == data
    s3.delete_file(key_bin)

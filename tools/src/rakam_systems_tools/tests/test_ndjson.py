"""
Tests for the NDJSON streaming helper (T1.2).

``iter_ndjson_lines`` is pure logic — unit-tested here with hand-built byte
iterables, no S3, no network. The ``stream_ndjson_from_s3`` roundtrip is a
critical-path integration test that skips cleanly without S3 credentials
(same pattern as test_s3.py).
"""
import json
import os
from datetime import datetime

import pytest

from rakam_systems_tools.utils.ndjson import (
    iter_ndjson_lines,
    stream_ndjson_from_s3,
)

# U+2028 (line separator) and U+2085 — str.splitlines() breaks on these, the
# byte-level split must not.
LS = " "
U2085 = "₅"


# ---- iter_ndjson_lines: pure unit tests ----

def test_splits_on_newline_byte_only():
    """A record whose string value embeds \\r, U+2028 and U+2085 survives whole.

    Regression guard for the str.splitlines() corruption bug.
    """
    value = f"line one\rline two{LS}line three{U2085}end"
    record = {"body": value}
    data = (json.dumps(record) + "\n").encode("utf-8")

    out = list(iter_ndjson_lines([data]))

    assert len(out) == 1
    line, _ = out[0]
    assert json.loads(line.decode("utf-8"))["body"] == value


def test_partial_line_buffered_across_chunks():
    """One logical record split across three arbitrary byte boundaries."""
    record = {"body": f"a{LS}b\rc", "n": 42}
    data = (json.dumps(record) + "\n").encode("utf-8")
    chunks = [data[:5], data[5:11], data[11:]]

    out = list(iter_ndjson_lines(chunks))

    assert len(out) == 1
    line, _ = out[0]
    assert json.loads(line.decode("utf-8")) == record


def test_end_offset_is_absolute_past_newline():
    """end_offset equals cumulative byte length including each newline."""
    r1 = b'{"a":1}'
    r2 = b'{"b":2}'
    r3 = b'{"c":3}'
    data = r1 + b"\n" + r2 + b"\n" + r3 + b"\n"

    out = list(iter_ndjson_lines([data]))

    offsets = [off for _, off in out]
    assert offsets == [
        len(r1) + 1,
        len(r1) + 1 + len(r2) + 1,
        len(r1) + 1 + len(r2) + 1 + len(r3) + 1,
    ]
    assert offsets[-1] == len(data)


def test_no_trailing_newline_yields_last_record():
    """Trailing bytes without a final newline are still yielded."""
    r1 = b'{"a":1}'
    r2 = b'{"b":2}'
    data = r1 + b"\n" + r2  # no final newline

    out = list(iter_ndjson_lines([data]))

    assert len(out) == 2
    assert out[1][0] == r2
    assert out[1][1] == len(data)


def test_blank_lines_skipped():
    """Embedded blank lines (\\n\\n) are not yielded, offsets stay absolute."""
    r1 = b'{"a":1}'
    r2 = b'{"b":2}'
    data = r1 + b"\n\n" + r2 + b"\n"

    out = list(iter_ndjson_lines([data]))

    lines = [line for line, _ in out]
    assert lines == [r1, r2]
    # r2's end_offset still counts the skipped blank line's newline.
    assert out[1][1] == len(data)


def test_skip_partial_first_line():
    """skip_partial_first_line drops the head; remaining offsets stay absolute."""
    partial = b'artial":1}'  # tail of a record split by a mid-object range read
    r2 = b'{"b":2}'
    r3 = b'{"c":3}'
    data = partial + b"\n" + r2 + b"\n" + r3 + b"\n"
    start = 100  # a ranged read began at byte 100

    out = list(
        iter_ndjson_lines(
            [data], start_offset=start, skip_partial_first_line=True
        )
    )

    lines = [line for line, _ in out]
    assert lines == [r2, r3]
    assert out[0][1] == start + len(partial) + 1 + len(r2) + 1
    assert out[1][1] == start + len(data)


# ---- stream_ndjson_from_s3: integration (skips without creds) ----

ONE_MIB = 1024 * 1024
TEST_PREFIX = "pytest_s3_test/"


def _s3_config_ok() -> bool:
    return bool(
        os.getenv("S3_ACCESS_KEY")
        and os.getenv("S3_SECRET_KEY")
        and os.getenv("S3_BUCKET_NAME")
    )


@pytest.fixture(scope="module")
def s3_ndjson_object():
    """Upload a >1 MiB NDJSON object; yield (key, records, size)."""
    if not _s3_config_ok():
        pytest.skip(
            "S3 tests require S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME"
        )
    from rakam_systems_tools.utils.s3 import s3

    # Enough records with a fat body to exceed 1 MiB and cross chunk boundaries.
    records = [
        {"id": i, "body": f"line{i}\rembedded{LS}sep{U2085} " + "x" * 200}
        for i in range(6000)
    ]
    payload = "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")
    assert len(payload) > ONE_MIB

    key = f"{TEST_PREFIX}ndjson_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson"
    s3.upload_file(key=key, content=payload,
                   content_type="application/x-ndjson")
    yield key, records, len(payload)
    s3.delete_file(key)


def test_stream_ndjson_from_s3_default_resume_keeps_boundary_record(monkeypatch):
    """Regression: resuming from a checkpoint ``end_offset`` (a clean boundary)
    must NOT drop the record that starts at that boundary. No S3 — the reader is
    faked, so this guards the wrapper's skip decision on every run.
    """
    from rakam_systems_tools.utils.s3 import s3 as s3mod

    data = b'{"a":1}\n{"b":2}\n{"c":3}\n'

    def fake_stream_file(key, bucket=None, chunk_size=1024, start_offset=0):
        yield data[start_offset:]

    monkeypatch.setattr(s3mod, "stream_file", fake_stream_file)

    full = list(stream_ndjson_from_s3("k"))
    assert [r for r, _ in full] == [{"a": 1}, {"b": 2}, {"c": 3}]

    # end_offset yielded after the first record is a clean boundary.
    checkpoint = full[0][1]
    resumed = list(stream_ndjson_from_s3("k", start_offset=checkpoint))
    assert [r for r, _ in resumed] == [{"b": 2}, {"c": 3}]  # {"b":2} NOT dropped
    assert resumed[-1][1] == len(data)


def test_stream_ndjson_from_s3_roundtrip_and_checkpoint_resume(s3_ndjson_object):
    """Full stream matches source; resuming from a checkpoint end_offset yields
    the next record onward without dropping the boundary record."""
    key, records, size = s3_ndjson_object

    streamed = list(stream_ndjson_from_s3(key))
    assert [r for r, _ in streamed] == records
    assert streamed[-1][1] == size

    # Resume from a real checkpoint: the end_offset yielded after record k.
    k = len(records) // 2
    checkpoint = streamed[k][1]
    resumed = list(stream_ndjson_from_s3(key, start_offset=checkpoint))
    assert [r for r, _ in resumed] == records[k + 1:]  # record k+1 not dropped
    assert resumed[-1][1] == size


def test_stream_ndjson_from_s3_arbitrary_offset_skips_partial(s3_ndjson_object):
    """With skip_partial_first_line=True, an arbitrary mid-record offset skips
    the shattered partial and starts on the next whole record."""
    key, records, size = s3_ndjson_object

    mid = size // 2
    resumed = list(
        stream_ndjson_from_s3(key, start_offset=mid, skip_partial_first_line=True)
    )
    assert resumed[0][0] in records
    resume_index = records.index(resumed[0][0])
    assert [r for r, _ in resumed] == records[resume_index:]
    assert resumed[-1][1] == size

"""
Byte-level NDJSON streaming for rakam_systems.

``iter_ndjson_lines`` turns a byte-chunk stream (e.g. from ``s3.stream_file``)
into NDJSON records, yielding ``(line_bytes, end_offset)`` where ``end_offset``
is the absolute byte offset past the record's terminating newline — the resume
checkpoint consumed by downstream ingestion stages.

The split is on the ``\n`` byte ONLY. ``str.splitlines()`` also breaks on
U+2028, U+2085 and ``\r``, which shatters records whose JSON string values
(email bodies) legitimately contain those characters. Doing the split at the
byte level is the canonical fix for that corruption.
"""
from __future__ import annotations

import json
from typing import Iterable, Iterator, Optional


def iter_ndjson_lines(
    chunks: Iterable[bytes],
    start_offset: int = 0,
    skip_partial_first_line: bool = False,
) -> Iterator[tuple[bytes, int]]:
    r"""Split a byte-chunk stream into NDJSON records on the ``\n`` byte only.

    Yields ``(line_bytes, end_offset)`` where ``end_offset`` is the absolute
    byte offset past the record's terminating newline — the resume checkpoint.

    - Splits on ``b"\n"`` ONLY (never ``str.splitlines()``); U+2028 / U+2085 /
      ``\r`` inside a JSON string value stay intact.
    - Carries a partial-line buffer across chunk boundaries.
    - Blank lines are skipped (not yielded).
    - Trailing bytes without a final newline are yielded as a record.
    - ``skip_partial_first_line=True`` drops the first (assumed partial) line —
      used with a mid-object Range resume so we start on a clean record boundary.

    Args:
        chunks: Any iterable of byte chunks (e.g. ``s3.stream_file(...)``).
        start_offset: Absolute byte offset the first chunk begins at, so the
            yielded ``end_offset`` values are absolute for a ranged resume.
        skip_partial_first_line: Drop everything up to and including the first
            newline, landing on a clean record boundary after a mid-object read.

    Yields:
        tuple[bytes, int]: ``(line_bytes, end_offset)`` per record.
    """
    buffer = b""
    offset = start_offset
    dropping = skip_partial_first_line

    for chunk in chunks:
        buffer += chunk
        # Split off every complete line; the remainder (no trailing \n yet)
        # stays in the buffer for the next chunk.
        while True:
            newline = buffer.find(b"\n")
            if newline == -1:
                break
            line = buffer[:newline]
            buffer = buffer[newline + 1:]
            offset += newline + 1
            if dropping:
                # First newline consumed — the partial head is behind us.
                dropping = False
                continue
            if line:
                yield line, offset

    # Trailing bytes with no final newline are a final record — unless we were
    # still waiting to drop the partial first line (the whole tail was it).
    if buffer and not dropping:
        offset += len(buffer)
        yield buffer, offset


def stream_ndjson_from_s3(
    key: str,
    bucket: Optional[str] = None,
    start_offset: int = 0,
) -> Iterator[tuple[dict, int]]:
    """Stream an S3 NDJSON object as ``(record_dict, end_offset)`` records.

    Wires T1.1's ``s3.stream_file`` into ``iter_ndjson_lines`` and JSON-decodes
    each record. When ``start_offset > 0``, issues the ranged read and drops the
    first partial line so decoding starts on a clean record boundary.

    Lives here (not in ``iter_ndjson_lines``) so JSON decoding stays out of the
    pure splitter and so ``rakam_systems_vectorstore`` never pulls in an S3
    dependency; the S3 import is lazy.

    Args:
        key: The S3 object key.
        bucket: Bucket name (defaults to ``S3_BUCKET_NAME``).
        start_offset: Absolute byte offset to resume from; 0 reads from the top.

    Yields:
        tuple[dict, int]: ``(record_dict, end_offset)`` per NDJSON record.
    """
    from rakam_systems_tools.utils.s3 import s3

    chunks = s3.stream_file(key, bucket=bucket, start_offset=start_offset)
    for line, end_offset in iter_ndjson_lines(
        chunks,
        start_offset=start_offset,
        skip_partial_first_line=start_offset > 0,
    ):
        yield json.loads(line.decode("utf-8")), end_offset


__all__ = [
    "iter_ndjson_lines",
    "stream_ndjson_from_s3",
]

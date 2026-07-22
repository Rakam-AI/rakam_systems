"""Tests for the standalone psycopg pgvector loader (T1.3a).

The DB write is a critical path, so it is exercised against a real
``pgvector/pgvector`` Postgres (integration over mocks per rakam policy),
gated on ``PGVECTOR_TEST_DSN``. Without that env var the integration tests
self-skip so collection succeeds anywhere. The vector-literal formatter is
pure logic and always runs.
"""
from __future__ import annotations

import os
import uuid

import pytest

from rakam_systems_vectorstore.components.loader.pgvector_loader import (
    PgVectorLoader,
    PgVectorLoaderConfig,
    _vector_literal,
)

_DSN = os.environ.get("PGVECTOR_TEST_DSN")
_requires_db = pytest.mark.skipif(
    not _DSN, reason="PGVECTOR_TEST_DSN not set — no pgvector Postgres available"
)


def test_vector_text_format():
    assert _vector_literal([0.1, 0.2]) == "[0.1,0.2]"
    assert _vector_literal([1.0, 2.0, 3.0, 4.0]) == "[1.0,2.0,3.0,4.0]"


def test_public_symbols_importable_from_package_root():
    import rakam_systems_vectorstore as vs

    assert vs.PgVectorLoader is PgVectorLoader
    assert vs.PgVectorLoaderConfig is PgVectorLoaderConfig


@pytest.fixture
def temp_table():
    """Create a fresh vector(4) table; drop it on teardown."""
    import psycopg

    table = f"test_pgvec_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE {table} ("
                "id serial PRIMARY KEY, content text, meta jsonb, embedding vector(4))"
            )
        conn.commit()
    yield table
    with psycopg.connect(_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()


def _fixture_rows_and_vectors(n: int):
    rows = [{"content": f"doc-{i}", "meta": '{"i": %d}' % i} for i in range(n)]
    vectors = [[float(i), float(i) + 0.1, float(i) + 0.2, float(i) + 0.3] for i in range(n)]
    return rows, vectors


@_requires_db
class TestPgVectorLoaderIntegration:
    def _config(self, table, **kw):
        return PgVectorLoaderConfig(
            dsn=_DSN, table=table, columns=["content", "meta"], **kw
        )

    def test_load_roundtrip(self, temp_table):
        import psycopg

        rows, vectors = _fixture_rows_and_vectors(1000)
        loader = PgVectorLoader(self._config(temp_table, batch_size=100))
        loader.load(rows, vectors)

        with psycopg.connect(_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {temp_table}")
                assert cur.fetchone()[0] == 1000
                cur.execute(
                    f"SELECT content, embedding FROM {temp_table} WHERE content = 'doc-7'"
                )
                content, embedding = cur.fetchone()
                assert content == "doc-7"
                assert str(embedding) == "[7,7.1,7.2,7.3]"

    def test_load_returns_count(self, temp_table):
        rows, vectors = _fixture_rows_and_vectors(250)
        loader = PgVectorLoader(self._config(temp_table, batch_size=100))
        assert loader.load(rows, vectors) == 250

    def test_load_is_batched_not_whole_corpus(self, temp_table, monkeypatch):
        rows, vectors = _fixture_rows_and_vectors(1000)
        loader = PgVectorLoader(self._config(temp_table, batch_size=100))
        calls = {"n": 0}
        original = loader._copy_batch

        def counting(cur, batch):
            calls["n"] += 1
            return original(cur, batch)

        monkeypatch.setattr(loader, "_copy_batch", counting)
        loader.load(rows, vectors)
        assert calls["n"] == 10  # 1000 / 100, never one giant statement

    def test_copy_and_executemany_parity(self, temp_table):
        import psycopg

        rows, vectors = _fixture_rows_and_vectors(200)
        PgVectorLoader(self._config(temp_table, use_copy=True)).load(rows, vectors)

        other = f"{temp_table}_em"
        with psycopg.connect(_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE TABLE {other} (LIKE {temp_table} INCLUDING ALL)"
                )
            conn.commit()
        try:
            rows2, vectors2 = _fixture_rows_and_vectors(200)
            PgVectorLoader(self._config(other, use_copy=False)).load(rows2, vectors2)
            with psycopg.connect(_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT count(*) FROM {temp_table}")
                    copy_count = cur.fetchone()[0]
                    cur.execute(f"SELECT count(*) FROM {other}")
                    em_count = cur.fetchone()[0]
                    assert copy_count == em_count == 200
                    cur.execute(
                        f"SELECT c.embedding = e.embedding "
                        f"FROM {temp_table} c JOIN {other} e USING (content)"
                    )
                    assert all(r[0] for r in cur.fetchall())
        finally:
            with psycopg.connect(_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {other}")
                conn.commit()

    def test_ensure_pgvector_extension_idempotent(self, temp_table):
        loader = PgVectorLoader(self._config(temp_table))
        loader.ensure_pgvector_extension()
        loader.ensure_pgvector_extension()  # no error on second call

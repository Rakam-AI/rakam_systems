"""Tests for the generic Neo4j loader stub (T1.3b).

Deferred-use (D13): a single connectivity+write integration test plus batching
unit coverage is enough — do not over-invest. Integration tests gate on
``NEO4J_TEST_URI`` / ``NEO4J_TEST_USER`` / ``NEO4J_TEST_PASSWORD`` and self-skip
otherwise so collection succeeds without a Neo4j server. The batching and
Cypher-guard tests are pure logic over a fake session and always run.
"""
from __future__ import annotations

import os

import pytest

from rakam_systems_vectorstore.components.loader.neo4j_loader import (
    Neo4jLoader,
    Neo4jLoaderConfig,
)

_URI = os.environ.get("NEO4J_TEST_URI")
_USER = os.environ.get("NEO4J_TEST_USER")
_PASSWORD = os.environ.get("NEO4J_TEST_PASSWORD")
_requires_db = pytest.mark.skipif(
    not (_URI and _USER and _PASSWORD),
    reason="NEO4J_TEST_* not set — no Neo4j server available",
)


class _FakeSession:
    def __init__(self, recorder):
        self._recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_write(self, fn):
        # execute_write(lambda tx, b=batch: tx.run(cypher, rows=b)) — capture the
        # batch the loader passed by running the callable against a fake tx.
        return fn(_FakeTx(self._recorder))


class _FakeTx:
    def __init__(self, recorder):
        self._recorder = recorder

    def run(self, cypher, rows):
        self._recorder.append(len(rows))
        return _FakeCursor()


class _FakeCursor:
    def consume(self):
        return None


class _FakeDriver:
    def __init__(self, recorder):
        self._recorder = recorder

    def session(self, database=None):
        return _FakeSession(self._recorder)


def _loader_with_fake_driver(recorder, batch_size):
    loader = object.__new__(Neo4jLoader)
    loader.config = Neo4jLoaderConfig(
        uri="bolt://x", user="u", password="p", batch_size=batch_size
    )
    loader._driver = _FakeDriver(recorder)
    return loader


def test_load_rejects_cypher_without_rows_param():
    recorder = []
    loader = _loader_with_fake_driver(recorder, batch_size=10)
    with pytest.raises(ValueError, match=r"\$rows"):
        loader.load("MERGE (n:Node {id: 1})", rows=[{"id": 1}])


def test_load_is_batched():
    recorder = []
    loader = _loader_with_fake_driver(recorder, batch_size=25)
    written = loader.load(
        "UNWIND $rows AS row MERGE (n:TestNode {id: row.id})",
        rows=[{"id": i} for i in range(100)],
    )
    assert written == 100
    assert recorder == [25, 25, 25, 25]  # 4 batched session writes


@_requires_db
class TestNeo4jLoaderIntegration:
    def _loader(self):
        return Neo4jLoader(
            Neo4jLoaderConfig(uri=_URI, user=_USER, password=_PASSWORD)
        )

    def test_verify_connectivity_ok(self):
        loader = self._loader()
        try:
            loader.verify_connectivity()  # no exception against live DB
        finally:
            loader.close()

    def test_load_writes_nodes(self):
        loader = self._loader()
        try:
            written = loader.load(
                "UNWIND $rows AS row MERGE (n:TestNode {id: row.id}) SET n.name = row.name",
                rows=[{"id": i, "name": f"n{i}"} for i in range(100)],
            )
            assert written == 100
            with loader._driver.session(database=loader.config.database) as session:
                count = session.run("MATCH (n:TestNode) RETURN count(n) AS c").single()["c"]
                assert count == 100
        finally:
            with loader._driver.session(database=loader.config.database) as session:
                session.run("MATCH (n:TestNode) DETACH DELETE n").consume()
            loader.close()

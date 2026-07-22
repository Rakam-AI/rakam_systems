"""Generic Neo4j loader stub (D13).

Deferred-use half of T1.3: we design the sink interface so a graph pipeline
*could* run on it, but do NOT migrate ``ots-graph-generation-service`` now. So
this is the minimal, contract-defining seam — a batched UNWIND writer over a
caller-supplied Cypher statement — not a re-home of the graph service.

Domain graph modelling (node labels, relationship types, MERGE keys) stays in
the caller's Cypher + params; this only manages the driver/session and batches
writes with bounded memory, mirroring PgVectorLoader's shape so the sink
interface is symmetric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


def _batched(rows: Iterator[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass
class Neo4jLoaderConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"
    batch_size: int = 500


class Neo4jLoader:
    """Generic batched writer over a parametrized Cypher statement."""

    def __init__(self, config: Neo4jLoaderConfig) -> None:
        from neo4j import GraphDatabase

        self.config = config
        self._driver = GraphDatabase.driver(
            config.uri, auth=(config.user, config.password)
        )

    def load(self, cypher: str, rows: Iterable[dict]) -> int:
        """Run ``cypher`` once per batch with ``UNWIND $rows AS row ...``.

        ``cypher`` MUST reference the ``$rows`` param — it is the batch payload.
        Streams ``rows`` in ``batch_size`` groups; returns rows written.
        """
        if "$rows" not in cypher:
            raise ValueError(
                "cypher must reference the $rows parameter "
                "(e.g. 'UNWIND $rows AS row ...') — it is the batch payload"
            )

        total = 0
        with self._driver.session(database=self.config.database) as session:
            for batch in _batched(iter(rows), self.config.batch_size):
                session.execute_write(
                    lambda tx, b=batch: tx.run(cypher, rows=b).consume()
                )
                total += len(batch)
        return total

    def verify_connectivity(self) -> None:
        """``driver.verify_connectivity()`` passthrough — the connectivity proof."""
        self._driver.verify_connectivity()

    def close(self) -> None:
        self._driver.close()

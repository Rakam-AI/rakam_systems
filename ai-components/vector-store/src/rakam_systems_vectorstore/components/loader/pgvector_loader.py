"""Standalone, Django-free pgvector loader for the ingestion job.

The library's ``ConfigurablePgVectorStore`` / ``PgVectorStore`` are bound to the
Django ORM (``django.db.connection``, ``NodeEntry.objects.bulk_create``) and need
a configured Django app to run. The ephemeral ingestion job (one-shot container /
CI ``workflow_dispatch``) has no Django project and must not acquire one, so this
reuses the *SQL insert shape* proven in ``ConfigurablePgVectorStore.add`` — the
``content, embedding, ...custom_metadata::jsonb`` columns and the ``[..]`` vector
literal — but issues it over plain ``psycopg`` (v3) so it runs anywhere.

Domain-agnostic on purpose (D5): the seed stage owns the ticket-rag column
mapping and hands this loader ``columns`` + per-row dicts. This module knows only
universal nouns — table, columns, rows, vectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator


def _vector_literal(vector: list[float]) -> str:
    """Render a pgvector text literal: ``[0.1,0.2]`` for ``[0.1, 0.2]``.

    pgvector accepts the same bracketed form on both the COPY text stream and a
    ``%s::vector`` insert param, so one formatter serves both paths.
    """
    return "[" + ",".join(map(str, vector)) + "]"


def _batched(pairs: Iterator[tuple[dict, list[float]]], size: int) -> Iterator[list[tuple[dict, list[float]]]]:
    batch: list[tuple[dict, list[float]]] = []
    for pair in pairs:
        batch.append(pair)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass
class PgVectorLoaderConfig:
    dsn: str
    table: str
    columns: list[str] = field(default_factory=list)
    vector_column: str = "embedding"
    batch_size: int = 500
    use_copy: bool = True


class PgVectorLoader:
    """Batched, bounded-memory writer into a pgvector table over psycopg v3.

    Peak memory is ~``batch_size`` rows, never O(corpus): rows and vectors are
    consumed as streams and committed per batch, so a crashed ephemeral job
    leaves a consistent committed prefix.
    """

    def __init__(self, config: PgVectorLoaderConfig) -> None:
        self.config = config

    def ensure_pgvector_extension(self) -> None:
        """``CREATE EXTENSION IF NOT EXISTS vector`` — idempotent, opt-in."""
        import psycopg

        with psycopg.connect(self.config.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

    def load(self, rows: Iterable[dict], vectors: Iterable[list[float]]) -> int:
        """Stream ``(row, vector)`` pairs into the target table in batches.

        ``rows[i][col]`` supplies each non-vector column named in
        ``config.columns``; ``vectors[i]`` supplies ``config.vector_column``.
        Returns the total number of rows written.
        """
        import psycopg

        cfg = self.config
        pairs = zip(rows, vectors)
        total = 0
        with psycopg.connect(cfg.dsn) as conn:
            for batch in _batched(iter(pairs), cfg.batch_size):
                with conn.cursor() as cur:
                    if cfg.use_copy:
                        self._copy_batch(cur, batch)
                    else:
                        self._insert_batch(cur, batch)
                conn.commit()
                total += len(batch)
        return total

    @property
    def _all_columns(self) -> list[str]:
        return [*self.config.columns, self.config.vector_column]

    def _copy_batch(self, cur, batch: list[tuple[dict, list[float]]]) -> None:
        cfg = self.config
        col_list = ", ".join(self._all_columns)
        with cur.copy(f"COPY {cfg.table} ({col_list}) FROM STDIN") as copy:
            for row, vector in batch:
                values = [row[c] for c in cfg.columns]
                values.append(_vector_literal(vector))
                copy.write_row(values)

    def _insert_batch(self, cur, batch: list[tuple[dict, list[float]]]) -> None:
        cfg = self.config
        col_list = ", ".join(self._all_columns)
        placeholders = ", ".join(["%s"] * len(cfg.columns)) + ", %s::vector"
        sql = f"INSERT INTO {cfg.table} ({col_list}) VALUES ({placeholders})"
        params = [
            [*(row[c] for c in cfg.columns), _vector_literal(vector)]
            for row, vector in batch
        ]
        cur.executemany(sql, params)

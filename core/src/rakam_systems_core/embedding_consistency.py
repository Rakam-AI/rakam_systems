"""Ingestion<->runtime embedding consistency.

An index/corpus is embedded once at ingestion; if it is later queried with a
different embedding model or dimension, retrieval degrades silently. To prevent
that, record an :class:`EmbeddingFingerprint` alongside each index at build time
and call :func:`validate_embedding_consistency` at startup.

Compatibility is keyed on ``(model_name, dim)`` and deliberately **ignores the
provider**: the same model served by a different provider (e.g.
``text-embedding-3-small`` on Azure vs OpenAI) yields identical vectors, so a
provider switch alone is fine. A different model or dimension is not.

Persistence is the caller's job (a ticket_indices row, a graph property, ...) --
this module only defines the fingerprint and the comparison.
"""
from __future__ import annotations

from typing import Mapping, Optional, Union

from pydantic import BaseModel, Field

from .config_schema import EmbeddingRef


class EmbeddingConsistencyError(ValueError):
    """Runtime embedding config is incompatible with the index it queries."""


class EmbeddingFingerprint(BaseModel):
    """Identity of the embedder that built an index, stored alongside it."""

    # model_name starts with model_ -> silence pydantic's protected-namespace warning.
    model_config = {"protected_namespaces": ()}

    model_name: str = Field(..., description="Model portion of the ref that built the index")
    dim: int = Field(..., gt=0, description="Vector dimension of the index")

    @classmethod
    def from_ref(cls, ref: EmbeddingRef) -> "EmbeddingFingerprint":
        return cls(model_name=ref.model_name, dim=ref.dim)

    def as_dict(self) -> dict:
        """Plain dict for storing on an index record / graph property."""
        return {"model_name": self.model_name, "dim": self.dim}


def validate_embedding_consistency(
    expected: EmbeddingRef,
    stored: Union[EmbeddingFingerprint, Mapping, None],
) -> None:
    """Raise :class:`EmbeddingConsistencyError` if ``expected`` is incompatible
    with the ``stored`` fingerprint recorded when the index was built.

    ``stored`` may be an :class:`EmbeddingFingerprint`, a mapping with
    ``model_name``/``dim``, or ``None``. ``None`` means the index predates
    fingerprinting: we cannot prove a mismatch, so this passes -- callers should
    log a warning in that case.
    """
    if stored is None:
        return
    if isinstance(stored, Mapping):
        stored = EmbeddingFingerprint(**stored)

    want = EmbeddingFingerprint.from_ref(expected)
    if stored.model_name != want.model_name or stored.dim != want.dim:
        raise EmbeddingConsistencyError(
            "embedding config does not match the index it queries: index built with "
            f"model={stored.model_name!r} dim={stored.dim}, runtime configured "
            f"model={want.model_name!r} dim={want.dim}. Re-embed the corpus or fix "
            "the embedding ref."
        )

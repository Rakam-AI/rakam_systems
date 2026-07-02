"""AI gateway embedder: build an ``EmbeddingModel`` from a standardized ref.

The vector stores are coded against the sync ``EmbeddingModel.run(texts)``
contract. pydantic-ai's ``Embedder`` exposes ``embed_documents_sync``, so the
adapter is a straight sync-to-sync wrapper -- no async bridge needed.

Routing:
- a local ``sentence-transformers`` ref uses the offline
  ``ConfigurableEmbeddings`` backend (pydantic-ai has no local ST embedder);
- everything else resolves through pydantic-ai (any OpenAI-compatible /
  voyage / google / cohere provider) and is wrapped in ``GatewayEmbeddings``.
"""
from __future__ import annotations

from typing import Any, List

from rakam_systems_core.config_schema import EmbeddingRef
from rakam_systems_core.interfaces.embedding_model import EmbeddingModel

# Provider prefixes that mean "run locally, offline" rather than call a provider.
_LOCAL_PROVIDERS = {"sentence-transformers", "sentence_transformer", "st", "local"}


class GatewayEmbeddings(EmbeddingModel):
    """``EmbeddingModel`` adapter over a pydantic-ai embedder.

    pydantic-ai's ``Embedder`` is async-native, so :meth:`arun` awaits
    ``embed_documents`` directly — the right path for async services (their
    handlers run inside an event loop, where the sync ``embed_documents_sync``
    would raise "event loop already running"). :meth:`run` stays for the sync
    ``EmbeddingModel`` contract (the vector store's own sync/threaded indexing).

    ``embedder`` needs ``embed_documents_sync`` and async ``embed_documents``,
    each returning a result whose ``.embeddings`` is a sequence of vectors; a
    stub with those is enough to unit-test the adapter.
    """

    def __init__(self, embedder: Any, dim: int, name: str = "gateway_embeddings") -> None:
        super().__init__(name=name)
        self._embedder = embedder
        self._dim = dim

    def _to_vectors(self, result: Any) -> List[List[float]]:
        vectors = [list(v) for v in result.embeddings]
        # Cheap invariant: the provider honoured the requested dimension. Guards
        # against a model/ref that silently returns a different-width vector,
        # which would corrupt the index. Full index<->runtime check is in the
        # embedding_consistency helper.
        if vectors and len(vectors[0]) != self._dim:
            raise ValueError(
                f"embedding dimension mismatch: model returned {len(vectors[0])}, "
                f"config expects {self._dim} (check the ref and dim)"
            )
        return vectors

    def run(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._to_vectors(self._embedder.embed_documents_sync(texts))

    async def arun(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._to_vectors(await self._embedder.embed_documents(texts))


def build_embedder(cfg: EmbeddingRef) -> EmbeddingModel:
    """Build an ``EmbeddingModel`` from a standardized :class:`EmbeddingRef`.

    Returns a live embedder ready for the vector stores' ``run(texts)`` call.
    pydantic-ai imports are lazy so importing this module never requires
    pydantic-ai unless the gateway path is actually used.
    """
    if cfg.provider in _LOCAL_PROVIDERS:
        from rakam_systems_vectorstore.components.embedding_model.configurable_embeddings import (
            ConfigurableEmbeddings,
        )

        return ConfigurableEmbeddings(
            config={"model_type": "sentence_transformer", "model_name": cfg.model_name}
        )

    from pydantic_ai.embeddings import Embedder, infer_embedding_model
    from pydantic_ai.embeddings.settings import EmbeddingSettings

    if cfg.base_url:
        # OpenAI-compatible endpoint (Ollama / local / compatible Azure).
        from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
        from pydantic_ai.providers.openai import OpenAIProvider

        model = OpenAIEmbeddingModel(
            cfg.model_name, provider=OpenAIProvider(base_url=cfg.base_url)
        )
    else:
        model = infer_embedding_model(cfg.ref)

    # dimensions is applied to every request so the output width equals cfg.dim
    # (this is what produces the graph's 384-dim truncation of text-embedding-3-*).
    embedder = Embedder(model, settings=EmbeddingSettings(dimensions=cfg.dim))
    return GatewayEmbeddings(embedder, dim=cfg.dim)

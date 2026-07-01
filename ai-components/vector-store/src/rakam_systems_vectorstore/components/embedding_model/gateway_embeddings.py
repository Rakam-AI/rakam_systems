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
    """Sync ``EmbeddingModel`` adapter over a pydantic-ai embedder.

    ``embedder`` only needs an ``embed_documents_sync(texts) -> result`` method
    whose result exposes ``.embeddings`` (a sequence of vectors); this keeps the
    adapter unit-testable with a stub and independent of pydantic-ai internals.
    """

    def __init__(self, embedder: Any, dim: int, name: str = "gateway_embeddings") -> None:
        super().__init__(name=name)
        self._embedder = embedder
        self._dim = dim

    def run(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = self._embedder.embed_documents_sync(texts)
        vectors = [list(v) for v in result.embeddings]
        # Cheap invariant: the provider honoured the requested dimension. Guards
        # against a model/ref that silently returns a different-width vector,
        # which would corrupt the index. Full index<->runtime check is PR 4.
        if vectors and len(vectors[0]) != self._dim:
            raise ValueError(
                f"embedding dimension mismatch: model returned {len(vectors[0])}, "
                f"config expects {self._dim} (check the ref and dim)"
            )
        return vectors


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

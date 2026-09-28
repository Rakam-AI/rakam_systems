"""AI gateway embedder: build an ``EmbeddingModel`` from a standardized ref.

The vector stores are coded against the sync ``EmbeddingModel.run(texts)``
contract. pydantic-ai's ``Embedder`` exposes ``embed_documents_sync``, so the
adapter is a straight sync-to-sync wrapper -- no async bridge needed.

Routing. Everything resolves through pydantic-ai except one enumerated exception
list: providers for which pydantic-ai ships **no embeddings backend at all**, so
that the alternative is not "a different SDK" but ``UserError: Unknown embeddings
model``. This list is the normative home of that exception (see the guardrail in
docs/specs/ai-gateway.md) and it has exactly two entries:

- ``sentence-transformers`` and its aliases -> the offline
  ``ConfigurableEmbeddings`` backend;
- ``mistral`` -> :class:`MistralEmbeddingModel`, this package's own
  ``pydantic_ai.embeddings`` backend.

Everything else -- any OpenAI-compatible / voyage / google / cohere / bedrock
provider -- goes to ``infer_embedding_model`` untouched. Admitting a provider here
requires showing that ``pydantic_ai/embeddings/`` ships no module for it in the
current release; removing one is mandatory the release after upstream ships a
backend. A provider pydantic-ai *does* support must never appear here.
"""
from __future__ import annotations

from typing import Any, List

from rakam_systems_core.config_schema import EmbeddingRef
from rakam_systems_core.interfaces.embedding_model import EmbeddingModel

# Provider prefixes that mean "run locally, offline" rather than call a provider.
_LOCAL_PROVIDERS = {"sentence-transformers", "sentence_transformer", "st", "local"}


def _mistral_server_origin(base_url: str) -> str:
    """Turn an OpenAI-style base URL into mistralai's origin-only ``server_url``.

    openai-python's ``base_url`` must include ``/v1``; mistralai's ``server_url``
    is the ORIGIN and the SDK appends ``/v1/embeddings`` itself. So an
    un-normalised ``https://host/v1`` would request ``https://host/v1/v1/embeddings``
    -- a 404 at request time, not a configuration error anyone sees at startup.
    ``EmbeddingRef.base_url`` is one field shared by both branches, so the
    difference is absorbed here rather than forked into the config convention.
    Only an exact trailing ``/v1`` segment is stripped, so a path-mounted proxy
    such as ``https://proxy/v1beta`` is left alone.
    """
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed


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

    def __init__(
        self,
        embedder: Any,
        dim: int,
        name: str = "gateway_embeddings",
        batch_size: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self._embedder = embedder
        self._dim = dim
        # None -> hand all texts to the provider in one call (original behavior,
        # what the 2 existing consumers get). A positive int chunks the corpus so
        # the embed stage stays under provider per-request limits and O(batch)
        # memory. Batches run in input order so vectors stay aligned with the
        # caller's rows (the pgvector loader zips them positionally).
        self._batch_size = batch_size

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
        if not self._batch_size:
            return self._to_vectors(self._embedder.embed_documents_sync(texts))
        out: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(
                self._to_vectors(
                    self._embedder.embed_documents_sync(texts[i : i + self._batch_size])
                )
            )
        return out

    async def arun(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self._batch_size:
            return self._to_vectors(await self._embedder.embed_documents(texts))
        out: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(
                self._to_vectors(
                    await self._embedder.embed_documents(texts[i : i + self._batch_size])
                )
            )
        return out


def build_embedder(cfg: EmbeddingRef, batch_size: int | None = None) -> EmbeddingModel:
    """Build an ``EmbeddingModel`` from a standardized :class:`EmbeddingRef`.

    Returns a live embedder ready for the vector stores' ``run(texts)`` call.
    pydantic-ai imports are lazy so importing this module never requires
    pydantic-ai unless the gateway path is actually used.

    ``batch_size`` defaults to ``None`` — a single provider call, the behavior
    the 2 existing product consumers rely on. The ingestion embed stage passes a
    positive size to bound per-request payload/memory over a large corpus.
    (``EmbeddingRef`` carries no batch field, so the param is the sole source.)
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

    if cfg.provider == "mistral":
        # pydantic-ai ships no pydantic_ai/embeddings/mistral.py -- verified absent
        # in every release through 2.37 -- so infer_embedding_model("mistral:...")
        # raises UserError: Unknown embeddings model. Second and last entry in the
        # "no upstream backend" exception list; see the module docstring. Delete
        # this branch the release after upstream ships a backend.
        #
        # MUST stay above the cfg.base_url branch: that branch is OpenAI-shaped, so
        # a mistral: ref carrying a base_url used to build an OpenAIEmbeddingModel
        # that authenticated with OPENAI_API_KEY and spoke the OpenAI wire format.
        from rakam_systems_vectorstore.components.embedding_model.mistral_embedding_model import (
            MistralEmbeddingModel,
        )

        model = MistralEmbeddingModel(
            cfg.model_name,
            base_url=_mistral_server_origin(cfg.base_url) if cfg.base_url else None,
        )
    elif cfg.base_url:
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
    return GatewayEmbeddings(embedder, dim=cfg.dim, batch_size=batch_size)

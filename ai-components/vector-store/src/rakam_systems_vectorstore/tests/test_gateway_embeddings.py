"""Tests for the AI gateway embedder (build_embedder / GatewayEmbeddings)."""
import asyncio

import pytest

from rakam_systems_core.config_schema import EmbeddingRef
from rakam_systems_vectorstore.components.embedding_model import gateway_embeddings
from rakam_systems_vectorstore.components.embedding_model.gateway_embeddings import (
    GatewayEmbeddings,
    build_embedder,
)


class _FakeResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeEmbedder:
    """Minimal stand-in for pydantic-ai's Embedder (sync + async paths)."""

    def __init__(self, embeddings):
        self._embeddings = embeddings
        self.calls = []
        self.async_calls = []

    def embed_documents_sync(self, texts):
        self.calls.append(list(texts))
        return _FakeResult(self._embeddings)

    async def embed_documents(self, texts):
        self.async_calls.append(list(texts))
        return _FakeResult(self._embeddings)


class TestGatewayEmbeddingsAdapter:
    def test_extracts_vectors_as_lists(self):
        fake = _FakeEmbedder([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        adapter = GatewayEmbeddings(fake, dim=4)
        out = adapter.run(["a", "b"])
        assert out == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        assert fake.calls == [["a", "b"]]

    def test_empty_input_short_circuits(self):
        fake = _FakeEmbedder([])
        assert GatewayEmbeddings(fake, dim=4).run([]) == []
        assert fake.calls == []  # embedder not called for empty input

    def test_dimension_mismatch_raises(self):
        fake = _FakeEmbedder([[0.1, 0.2, 0.3]])  # width 3
        with pytest.raises(ValueError, match="dimension mismatch"):
            GatewayEmbeddings(fake, dim=4).run(["a"])


class TestGatewayEmbeddingsAsync:
    def test_arun_awaits_native_async_path(self):
        fake = _FakeEmbedder([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        adapter = GatewayEmbeddings(fake, dim=4)
        # arun must work *inside* a running loop — the whole reason it exists
        # (embed_documents_sync would raise "event loop already running" here).
        out = asyncio.run(adapter.arun(["a", "b"]))
        assert out == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        assert fake.async_calls == [["a", "b"]]  # took the async path, not sync
        assert fake.calls == []

    def test_arun_empty_short_circuits(self):
        fake = _FakeEmbedder([])
        assert asyncio.run(GatewayEmbeddings(fake, dim=4).arun([])) == []
        assert fake.async_calls == []

    def test_arun_dimension_mismatch_raises(self):
        fake = _FakeEmbedder([[0.1, 0.2, 0.3]])  # width 3
        with pytest.raises(ValueError, match="dimension mismatch"):
            asyncio.run(GatewayEmbeddings(fake, dim=4).arun(["a"]))


class TestPublicSurface:
    def test_build_embedder_and_gateway_embeddings_importable_from_package_root(self):
        import rakam_systems_vectorstore as vs

        assert vs.build_embedder is build_embedder
        assert vs.GatewayEmbeddings is GatewayEmbeddings


class TestBuildEmbedder:
    @pytest.fixture(autouse=True)
    def _dummy_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def test_openai_ref_returns_adapter(self):
        emb = build_embedder(EmbeddingRef(ref="openai:text-embedding-3-small", dim=1536))
        assert isinstance(emb, GatewayEmbeddings)
        assert emb._dim == 1536

    def test_base_url_ref_returns_adapter(self):
        emb = build_embedder(
            EmbeddingRef(
                ref="openai:nomic-embed-text",
                base_url="http://localhost:11434/v1",
                dim=768,
            )
        )
        assert isinstance(emb, GatewayEmbeddings)
        assert emb._dim == 768

    def test_local_sentence_transformers_routes_to_configurable(self, monkeypatch):
        # Avoid importing torch/sentence-transformers: stub ConfigurableEmbeddings.
        captured = {}

        class _StubConfigurable:
            def __init__(self, config=None):
                captured["config"] = config

        import rakam_systems_vectorstore.components.embedding_model.configurable_embeddings as ce
        monkeypatch.setattr(ce, "ConfigurableEmbeddings", _StubConfigurable)

        emb = build_embedder(EmbeddingRef(ref="sentence-transformers:all-MiniLM-L6-v2", dim=384))
        assert isinstance(emb, _StubConfigurable)
        assert captured["config"]["model_type"] == "sentence_transformer"
        assert captured["config"]["model_name"] == "all-MiniLM-L6-v2"

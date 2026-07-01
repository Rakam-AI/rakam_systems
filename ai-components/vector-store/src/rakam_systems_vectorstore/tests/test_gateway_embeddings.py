"""Tests for the AI gateway embedder (build_embedder / GatewayEmbeddings)."""
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
    """Minimal stand-in for pydantic-ai's Embedder (sync path only)."""

    def __init__(self, embeddings):
        self._embeddings = embeddings
        self.calls = []

    def embed_documents_sync(self, texts):
        self.calls.append(list(texts))
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

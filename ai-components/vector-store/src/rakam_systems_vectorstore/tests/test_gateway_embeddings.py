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


class _EchoEmbedder:
    """Returns one index-encoding vector per input text, so batch splitting and
    output order are both checkable. ``calls`` records each batch's size."""

    def __init__(self, dim=4):
        self._dim = dim
        self.calls = []
        self.async_calls = []

    def _vectors(self, texts):
        # Each text is "t{k}"; encode k in the first slot, pad to dim so the
        # per-batch dim-guard stays satisfied.
        return _FakeResult(
            [[float(int(t[1:]))] + [0.0] * (self._dim - 1) for t in texts]
        )

    def embed_documents_sync(self, texts):
        self.calls.append(len(texts))
        return self._vectors(texts)

    async def embed_documents(self, texts):
        self.async_calls.append(len(texts))
        return self._vectors(texts)


class TestGatewayEmbeddingsBatching:
    def test_batching_splits_into_expected_calls(self):
        fake = _EchoEmbedder()
        adapter = GatewayEmbeddings(fake, dim=4, batch_size=100)
        adapter.run([f"t{k}" for k in range(250)])
        assert fake.calls == [100, 100, 50]

    def test_batching_preserves_order_and_count(self):
        fake = _EchoEmbedder()
        adapter = GatewayEmbeddings(fake, dim=4, batch_size=100)
        out = adapter.run([f"t{k}" for k in range(250)])
        assert len(out) == 250
        assert [int(v[0]) for v in out] == list(range(250))  # vector k <- input k

    def test_batch_size_none_is_single_call(self):
        # Regression guard for the 2 existing consumers: default path is untouched.
        fake = _EchoEmbedder()
        adapter = GatewayEmbeddings(fake, dim=4)
        adapter.run([f"t{k}" for k in range(250)])
        assert fake.calls == [250]

    def test_arun_batches_in_order(self):
        fake = _EchoEmbedder()
        adapter = GatewayEmbeddings(fake, dim=4, batch_size=100)
        out = asyncio.run(adapter.arun([f"t{k}" for k in range(250)]))
        assert fake.async_calls == [100, 100, 50]
        assert [int(v[0]) for v in out] == list(range(250))
        assert fake.calls == []  # async path only


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
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    def test_openai_ref_returns_adapter(self):
        emb = build_embedder(EmbeddingRef(ref="openai:text-embedding-3-small", dim=1536))
        assert isinstance(emb, GatewayEmbeddings)
        assert emb._dim == 1536

    def test_build_embedder_threads_batch_size(self):
        cfg = EmbeddingRef(ref="openai:text-embedding-3-small", dim=1536)
        assert build_embedder(cfg, batch_size=64)._batch_size == 64
        assert build_embedder(cfg)._batch_size is None  # default unchanged

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


class TestMistralRouting:
    """`mistral:` refs get this package's own backend, not pydantic-ai's.

    pydantic-ai ships no embeddings backend for Mistral, so before this branch
    existed `mistral:mistral-embed` raised UserError and `mistral:` + base_url
    silently built an OpenAIEmbeddingModel authenticating with OPENAI_API_KEY.
    """

    @pytest.fixture(autouse=True)
    def _dummy_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    @staticmethod
    def _inner_model(adapter):
        # Embedder.model is public; the private _get_model() is not.
        return adapter._embedder.model

    def test_mistral_ref_builds_a_mistral_embedding_model(self):
        from rakam_systems_vectorstore.components.embedding_model.mistral_embedding_model import (
            MistralEmbeddingModel,
        )

        emb = build_embedder(EmbeddingRef(ref="mistral:mistral-embed", dim=1024))
        assert isinstance(emb, GatewayEmbeddings)
        assert emb._dim == 1024
        model = self._inner_model(emb)
        assert isinstance(model, MistralEmbeddingModel)
        assert model.system == "mistral"
        assert model.model_name == "mistral-embed"

    def test_mistral_ref_with_base_url_does_not_route_through_openai(self):
        # The regression test for the live mis-route: with both keys set, the
        # OpenAI-shaped branch would happily win and nothing would look wrong.
        from rakam_systems_vectorstore.components.embedding_model.mistral_embedding_model import (
            MistralEmbeddingModel,
        )

        emb = build_embedder(
            EmbeddingRef(
                ref="mistral:mistral-embed", base_url="https://gw.internal/v1", dim=8
            )
        )
        model = self._inner_model(emb)
        assert isinstance(model, MistralEmbeddingModel)
        # Normalised to the origin: the SDK appends /v1/embeddings itself, so an
        # unstripped /v1 would request /v1/v1/embeddings.
        assert model.base_url == "https://gw.internal"

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("https://h", "https://h"),
            ("https://h/", "https://h"),
            ("https://h/v1", "https://h"),
            ("https://h/v1/", "https://h"),
            ("https://p/mistral/v1", "https://p/mistral"),
            ("https://p/v1beta", "https://p/v1beta"),
        ],
    )
    def test_mistral_server_origin(self, given, expected):
        assert gateway_embeddings._mistral_server_origin(given) == expected

    def test_other_refs_are_unaffected(self):
        # The three pre-existing routes still land where they did.
        from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel

        plain = build_embedder(EmbeddingRef(ref="openai:text-embedding-3-small", dim=1536))
        assert isinstance(self._inner_model(plain), OpenAIEmbeddingModel)
        compat = build_embedder(
            EmbeddingRef(ref="openai:nomic-embed-text", base_url="http://localhost:11434/v1", dim=768)
        )
        assert isinstance(self._inner_model(compat), OpenAIEmbeddingModel)

    def test_mistral_embedder_runs_end_to_end_through_the_adapter(self):
        # Fake transport -> real Mistral SDK -> MistralEmbeddingModel -> real
        # Embedder -> real GatewayEmbeddings, covering both the sync and async
        # entry points the vector stores actually call.
        import httpx
        from mistralai import Mistral
        from pydantic_ai.embeddings import Embedder
        from pydantic_ai.embeddings.settings import EmbeddingSettings

        from rakam_systems_vectorstore.components.embedding_model.mistral_embedding_model import (
            MistralEmbeddingModel,
        )

        def handler(request):
            return httpx.Response(200, json={
                "id": "emb-1", "object": "list", "model": "mistral-embed",
                "usage": {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4},
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.5, 0.6, 0.7, 0.8]},
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]},
                ],
            })

        client = Mistral(
            api_key="sk-fake",
            async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        model = MistralEmbeddingModel("mistral-embed", client=client)
        adapter = GatewayEmbeddings(
            Embedder(model, settings=EmbeddingSettings(dimensions=4)), dim=4
        )
        expected = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        # run() uses embed_documents_sync, which drives its own loop -- so it must
        # be called from sync context, never from inside an async test.
        assert adapter.run(["a", "b"]) == expected
        assert asyncio.run(adapter.arun(["a", "b"])) == expected

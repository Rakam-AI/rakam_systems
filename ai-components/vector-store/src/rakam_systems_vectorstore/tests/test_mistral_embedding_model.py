"""Tests for the Mistral embeddings backend (MistralEmbeddingModel).

Hermetic: a real ``mistralai.Mistral`` is driven through an ``httpx.MockTransport``,
so the SDK's own request serialisation and response validation are exercised while
nothing leaves the process. That matters here -- most of what this class does is
translate between the SDK's shapes and pydantic-ai's, so faking the SDK away would
test the translation against our own guess of it.

Every test is a SYNC function calling ``asyncio.run(...)``. There is no
``asyncio_mode`` configured anywhere in this repo and pytest-asyncio defaults to
strict, so an unmarked ``async def test_`` silently fails to collect.
"""
import asyncio
import json

import httpx
import pytest
from mistralai import Mistral
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)

from rakam_systems_vectorstore.components.embedding_model.mistral_embedding_model import (
    MistralEmbeddingModel,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _response(data, *, prompt_tokens=11, model="mistral-embed", resp_id="emb-1"):
    return {
        "id": resp_id,
        "object": "list",
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        },
        "data": data,
    }


def _vec(index, embedding):
    return {"object": "embedding", "index": index, "embedding": embedding}


def _model(payload=None, *, status=200, raises=None, capture=None, model_name="mistral-embed"):
    """Build a MistralEmbeddingModel whose transport returns ``payload``.

    ``capture``, if given, is a dict that receives the request url and body.
    ``raises`` short-circuits the transport with a connection-level failure.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if raises is not None:
            raise raises
        if capture is not None:
            capture["url"] = str(request.url)
            capture["body"] = json.loads(request.content)
        return httpx.Response(status, json=payload)

    client = Mistral(
        api_key="sk-fake",
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return MistralEmbeddingModel(model_name, client=client)


def _embed(model, texts=("a", "b"), **kw):
    return asyncio.run(model.embed(list(texts), input_type="document", **kw))


class TestIdentity:
    def test_reports_name_system_and_default_base_url(self):
        model = _model(_response([_vec(0, [0.1, 0.2])]))
        assert model.model_name == "mistral-embed"
        assert model.system == "mistral"
        assert model.base_url == "https://api.mistral.ai"

    def test_configured_base_url_is_reported(self):
        model = MistralEmbeddingModel(
            "mistral-embed", api_key="sk-fake", base_url="https://gw.internal"
        )
        assert model.base_url == "https://gw.internal"

    def test_hosted_without_a_key_fails_at_construction(self, monkeypatch):
        # Mirrors the OpenAI route: openai-python raises at construction when it is
        # pointed at the hosted API with no key. mistralai would build happily and
        # 401 mid-ingestion instead, so the check is ours.
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="needs an API key"):
            MistralEmbeddingModel("mistral-embed")

    def test_a_base_url_allows_a_keyless_endpoint(self, monkeypatch):
        # The other half of that contract: a base_url means "my own endpoint", and
        # the OpenAI branch has always tolerated a missing key there. Self-hosted
        # and proxied Mistral deployments must keep working.
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        model = MistralEmbeddingModel("mistral-embed", base_url="https://gw.internal")
        assert model.base_url == "https://gw.internal"

    def test_client_and_build_arguments_together_are_rejected(self):
        # A prebuilt client already carries key/endpoint/transport; silently
        # dropping the others would be the wrong kind of quiet.
        with pytest.raises(ValueError, match="not both"):
            MistralEmbeddingModel(
                "mistral-embed", api_key="sk-fake", client=Mistral(api_key="k")
            )


class TestRequest:
    def test_dimensions_setting_becomes_output_dimension(self):
        cap = {}
        _embed(_model(_response([_vec(0, [0.1, 0.2]), _vec(1, [0.3, 0.4])]), capture=cap),
               settings={"dimensions": 2})
        assert cap["body"]["output_dimension"] == 2
        assert cap["body"]["input"] == ["a", "b"]

    def test_output_dimension_is_omitted_when_dimensions_is_unset(self):
        # mistralai defaults this to an Unset() sentinel, not None: passing None
        # through would serialise `"output_dimension": null` into the body.
        # build_embedder always injects a dim, so only a direct call catches this.
        cap = {}
        _embed(_model(_response([_vec(0, [0.1]), _vec(1, [0.2])]), capture=cap))
        assert "output_dimension" not in cap["body"]

    def test_extra_headers_setting_is_forwarded(self):
        cap = {}
        _embed(_model(_response([_vec(0, [0.1]), _vec(1, [0.2])]), capture=cap),
               settings={"extra_headers": {"x-trace": "abc"}})
        assert cap["url"].endswith("/v1/embeddings")

    def test_empty_input_makes_no_request(self):
        # The server 422s on an empty input list and Embedder does not guard it.
        def handler(request):  # pragma: no cover - must never run
            raise AssertionError("the SDK was called for an empty input list")

        client = Mistral(
            api_key="sk-fake",
            async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        result = asyncio.run(
            MistralEmbeddingModel("mistral-embed", client=client).embed(
                [], input_type="document"
            )
        )
        assert list(result.embeddings) == []


class TestResponseMapping:
    def test_prompt_tokens_are_reported_as_input_tokens(self):
        # The only defence against the silent-zero trap: RequestUsage.extract with
        # api_flavor="embeddings" swallows its own ValueError for the mistral
        # provider and returns an empty usage with no error and no warning.
        result = _embed(_model(_response([_vec(0, [0.1]), _vec(1, [0.2])], prompt_tokens=7)))
        assert result.usage.input_tokens == 7

    def test_missing_prompt_tokens_becomes_zero_not_none(self):
        payload = _response([_vec(0, [0.1])], prompt_tokens=None)
        result = _embed(_model(payload), texts=("a",))
        assert result.usage.input_tokens == 0

    def test_vectors_are_reordered_by_index(self):
        # GatewayEmbeddings hands the list straight to a positional zip in the
        # pgvector loader, so a permuted response would mis-attach every vector.
        result = _embed(_model(_response([_vec(1, [0.3, 0.4]), _vec(0, [0.1, 0.2])])))
        assert [list(v) for v in result.embeddings] == [[0.1, 0.2], [0.3, 0.4]]

    def test_arrival_order_is_kept_when_index_is_absent(self):
        payload = _response([
            {"object": "embedding", "embedding": [0.1, 0.2]},
            {"object": "embedding", "embedding": [0.3, 0.4]},
        ])
        result = _embed(_model(payload))
        assert [list(v) for v in result.embeddings] == [[0.1, 0.2], [0.3, 0.4]]

    def test_model_and_response_id_are_propagated(self):
        result = _embed(_model(_response([_vec(0, [0.1]), _vec(1, [0.2])],
                                         model="mistral-embed-2", resp_id="req-42")))
        assert result.model_name == "mistral-embed-2"
        assert result.provider_response_id == "req-42"
        assert result.provider_name == "mistral"
        assert result.input_type == "document"
        assert list(result.inputs) == ["a", "b"]


class TestResponseGuards:
    def test_duplicate_indexes_are_rejected(self):
        # sorted() is stable, so an all-zero index set would otherwise pass a plain
        # sort and yield a wrong-but-plausible order.
        payload = _response([_vec(0, [0.1]), _vec(0, [0.2])])
        with pytest.raises(UnexpectedModelBehavior, match="expected each of"):
            _embed(_model(payload))

    def test_non_contiguous_indexes_are_rejected(self):
        payload = _response([_vec(0, [0.1]), _vec(5, [0.2])])
        with pytest.raises(UnexpectedModelBehavior, match="expected each of"):
            _embed(_model(payload))

    def test_mixed_index_presence_is_rejected(self):
        payload = _response([
            _vec(0, [0.1]),
            {"object": "embedding", "embedding": [0.2]},
        ])
        with pytest.raises(UnexpectedModelBehavior, match="mix of indexed"):
            _embed(_model(payload))

    def test_a_short_response_is_rejected(self):
        with pytest.raises(UnexpectedModelBehavior, match="for 2 inputs"):
            _embed(_model(_response([_vec(0, [0.1])])))

    def test_a_null_embedding_is_rejected(self):
        payload = _response([
            _vec(0, [0.1]),
            {"object": "embedding", "index": 1, "embedding": None},
        ])
        with pytest.raises(UnexpectedModelBehavior, match="null embedding"):
            _embed(_model(payload))


class TestErrorMapping:
    def test_http_error_becomes_model_http_error(self):
        with pytest.raises(ModelHTTPError) as exc:
            _embed(_model({"message": "rate limited"}, status=429))
        assert exc.value.status_code == 429

    def test_validation_error_becomes_model_http_error(self):
        # 422 arrives as HTTPValidationError, a sibling of SDKError under
        # MistralError; pydantic-ai's own models/mistral.py lets this one escape.
        payload = {"detail": [{"loc": ["body"], "msg": "bad", "type": "value_error"}]}
        with pytest.raises(ModelHTTPError) as exc:
            _embed(_model(payload, status=422))
        assert exc.value.status_code == 422

    def test_transport_error_becomes_model_api_error(self):
        # mistralai does not wrap connection failures, so without our clause a raw
        # httpx error would surface from inside a pydantic-ai call stack.
        with pytest.raises(ModelAPIError):
            _embed(_model(None, raises=httpx.ConnectError("no route")))

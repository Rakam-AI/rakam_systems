"""Tests for the raw-SDK MistralGateway.

There was no direct test of this class before -- only a factory test with the
whole gateway mocked out -- which is how a silently-discarded `base_url`, a
structured-output path that could not express a numeric bound, and `len // 4`
token counting all survived.

Hermetic: a real `mistralai.Mistral` driven through an `httpx.MockTransport`, so
the SDK's own request serialisation is exercised. Nothing leaves the process.
"""

import httpx
import pytest

pytest.importorskip("mistralai")

from mistralai import Mistral  # noqa: E402

from rakam_systems_core.interfaces.llm_gateway import LLMRequest  # noqa: E402
from rakam_systems_agent.components.llm_gateway.mistral_gateway import (  # noqa: E402
    MistralGateway,
    _server_origin,
)


@pytest.fixture(autouse=True)
def _dummy_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


def _completion(content='{"answer": "ok"}'):
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "mistral-small-latest",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _wire(gateway, handler):
    """Swap in a transport-mocked client, preserving the configured endpoint."""
    server_url = gateway.client.sdk_configuration.get_server_details()[0]
    gateway.client = Mistral(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        server_url=server_url,
    )
    return gateway


class TestEndpoint:
    def test_default_endpoint_is_unchanged(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json=_completion())

        gw = _wire(MistralGateway(model="mistral-small-latest"), handler)
        gw.generate(LLMRequest(user_prompt="hi"))
        assert seen["url"] == "https://api.mistral.ai/v1/chat/completions"
        assert gw.base_url is None

    @pytest.mark.parametrize(
        "given",
        ["https://gw.example", "https://gw.example/", "https://gw.example/v1", "https://gw.example/v1/"],
    )
    def test_base_url_is_normalized(self, given):
        # mistralai appends /v1/chat/completions to server_url itself, so an
        # unstripped /v1 would request /v1/v1/chat/completions -- a 404 nobody
        # sees until call time.
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json=_completion())

        gw = _wire(MistralGateway(model="mistral-small-latest", base_url=given), handler)
        gw.generate(LLMRequest(user_prompt="hi"))
        assert seen["url"] == "https://gw.example/v1/chat/completions"

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("https://h", "https://h"),
            ("https://h/v1", "https://h"),
            ("https://p/mistral/v1", "https://p/mistral"),
            ("https://p/v1beta", "https://p/v1beta"),
        ],
    )
    def test_server_origin(self, given, expected):
        assert _server_origin(given) == expected


class TestGenerate:
    def test_generate_returns_content_and_usage(self):
        gw = _wire(
            MistralGateway(model="mistral-small-latest"),
            lambda request: httpx.Response(200, json=_completion(content="hello")),
        )
        response = gw.generate(LLMRequest(user_prompt="hi"))
        assert response.content == "hello"
        assert response.usage["total_tokens"] == 5
        assert response.finish_reason == "stop"

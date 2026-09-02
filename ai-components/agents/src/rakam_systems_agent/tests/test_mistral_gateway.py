"""Tests for the raw-SDK MistralGateway.

There was no direct test of this class before -- only a factory test with the
whole gateway mocked out -- which is how a silently-discarded `base_url`, a
structured-output path that could not express a numeric bound, and `len // 4`
token counting all survived.

Hermetic: a real `mistralai.Mistral` driven through an `httpx.MockTransport`, so
the SDK's own request serialisation is exercised. Nothing leaves the process.
"""

import json

import httpx
import pytest
from pydantic import BaseModel, Field

pytest.importorskip("mistralai")

from mistralai import Mistral  # noqa: E402

from rakam_systems_core.interfaces.llm_gateway import LLMRequest  # noqa: E402
from rakam_systems_agent.components.llm_gateway.mistral_gateway import (  # noqa: E402
    MistralGateway,
    _server_origin,
    _strict_json_schema,
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


class Answer(BaseModel):
    answer: str


class HardSchema(BaseModel):
    """The exact shape mistralai's own response_format helper cannot convert."""

    answer: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    tags: dict[str, str] = {}
    note: str | None = None


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


class TestStrictJsonSchema:
    def test_numeric_defaults_and_bounds_survive(self):
        # mistralai's own rec_strict_json_schema raises ValueError: Unexpected
        # type: 0.5 here, which is why this helper exists.
        out = _strict_json_schema(HardSchema.model_json_schema())
        conf = out["properties"]["confidence"]
        assert conf["default"] == 0.5 and conf["minimum"] == 0.0 and conf["maximum"] == 1.0

    def test_object_nodes_are_closed(self):
        out = _strict_json_schema(Answer.model_json_schema())
        assert out["additionalProperties"] is False

    def test_dict_fields_keep_their_value_schema(self):
        # Rewriting additionalProperties to False here would make the field
        # uninhabitable -- no key could ever validate.
        out = _strict_json_schema(HardSchema.model_json_schema())
        assert out["properties"]["tags"]["additionalProperties"] == {"type": "string"}

    def test_optional_fields_are_not_promoted_to_required(self):
        # Synthesising required=list(properties) would silently rewrite the
        # caller's contract and disagree with the json_object fallback.
        out = _strict_json_schema(HardSchema.model_json_schema())
        assert out["required"] == ["answer"]

    def test_a_field_literally_named_properties_is_not_mistaken_for_a_container(self):
        class Weird(BaseModel):
            properties: str

        out = _strict_json_schema(Weird.model_json_schema())
        assert out["properties"]["properties"]["type"] == "string"


class TestStructuredOutput:
    def test_native_mode_sends_a_strict_json_schema(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json=_completion('{"answer":"yes","confidence":0.9,"tags":{},"note":null}'),
            )

        gw = _wire(MistralGateway(model="mistral-small-latest"), handler)
        result = gw.generate_structured(LLMRequest(user_prompt="q"), HardSchema)
        rf = seen["body"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "HardSchema"
        assert rf["json_schema"]["strict"] is True
        assert result.confidence == 0.9

    def test_json_object_mode_reproduces_the_original_request(self):
        # The opt-out for anyone depending on the old wire format byte for byte.
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_completion())

        gw = _wire(
            MistralGateway(model="mistral-small-latest", structured_mode="json_object"),
            handler,
        )
        gw.generate_structured(LLMRequest(user_prompt="q"), Answer)
        assert seen["body"]["response_format"] == {"type": "json_object"}
        # The schema travels in the system prompt, as it always did.
        assert "valid JSON that matches this schema" in seen["body"]["messages"][0]["content"]

    def test_auto_downgrades_once_and_stays_downgraded(self):
        formats = []

        def handler(request):
            body = json.loads(request.content)
            fmt = body["response_format"]["type"]
            formats.append(fmt)
            if fmt == "json_schema":
                return httpx.Response(
                    422,
                    json={"detail": [{"loc": ["body"], "msg": "unsupported", "type": "value_error"}]},
                )
            return httpx.Response(200, json=_completion())

        gw = _wire(MistralGateway(model="mistral-small-latest"), handler)
        assert gw.generate_structured(LLMRequest(user_prompt="q"), Answer).answer == "ok"
        assert gw.generate_structured(LLMRequest(user_prompt="q"), Answer).answer == "ok"
        # Probed once, then never again -- a rejection costs one extra request per
        # gateway, not one per call.
        assert formats == ["json_schema", "json_object", "json_object"]
        assert gw._structured_downgraded is True

    def test_json_schema_mode_raises_instead_of_downgrading(self):
        def handler(request):
            return httpx.Response(422, json={"detail": []})

        gw = _wire(
            MistralGateway(model="mistral-small-latest", structured_mode="json_schema"),
            handler,
        )
        with pytest.raises(Exception):
            gw.generate_structured(LLMRequest(user_prompt="q"), Answer)

    def test_a_null_content_response_is_rejected_readably(self):
        # AssistantMessage.content is nullable; model_validate_json(None) would
        # raise something that names neither the gateway nor the schema.
        def handler(request):
            return httpx.Response(200, json=_completion(content=None))

        gw = _wire(
            MistralGateway(model="mistral-small-latest", structured_mode="json_schema"),
            handler,
        )
        with pytest.raises(ValueError, match="no text content"):
            gw.generate_structured(LLMRequest(user_prompt="q"), Answer)


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

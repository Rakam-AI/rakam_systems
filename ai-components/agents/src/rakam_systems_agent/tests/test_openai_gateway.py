"""Tests for the raw-SDK OpenAIGateway.

Only the usage-omitted regression for now: there was no direct test of this class,
which is how a logging line that crashes a successful call survived.
"""
import httpx
import pytest

pytest.importorskip("openai")

from openai import OpenAI  # noqa: E402

from rakam_systems_core.interfaces.llm_gateway import LLMRequest  # noqa: E402
from rakam_systems_agent.components.llm_gateway.openai_gateway import (  # noqa: E402
    OpenAIGateway,
)


@pytest.fixture(autouse=True)
def _dummy_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_generate_returns_a_response_when_the_api_omits_usage():
    # A valid completion with no `usage` block. The gateway already tolerates this
    # when building LLMResponse; the info log used to dereference it anyway, so a
    # good answer was destroyed between construction and return.
    def handler(request):
        return httpx.Response(200, json={
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
        })

    gateway = OpenAIGateway(model="gpt-4o")
    gateway.client = OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = gateway.generate(LLMRequest(user_prompt="hi"))
    assert response.content == "hello"
    assert response.usage is None


def test_generate_reports_usage_when_present():
    def handler(request):
        return httpx.Response(200, json={
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        })

    gateway = OpenAIGateway(model="gpt-4o")
    gateway.client = OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert gateway.generate(LLMRequest(user_prompt="hi")).usage["total_tokens"] == 5

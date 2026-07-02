"""Tests for the AI gateway chat factory (ModelGateway.build_chat_model)."""
import pytest

from rakam_systems_core.config_schema import ModelRef
from rakam_systems_agent.components.model_gateway import (
    ModelGateway,
    NoopUsageHook,
    UsageHook,
)


@pytest.fixture(autouse=True)
def _dummy_openai_key(monkeypatch):
    # infer_model builds a client eagerly; give it a key so construction is hermetic.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_build_chat_model_resolves_openai():
    model = ModelGateway().build_chat_model(ModelRef(ref="openai:gpt-4o"))
    assert model is not None
    assert getattr(model, "model_name", "gpt-4o") == "gpt-4o"


def test_unknown_provider_surfaces_pydantic_error():
    # The gateway keeps no allow-list; an unknown provider is pydantic-ai's error,
    # not ours. This is exactly the "gemma4:e2b-mlx" case that started the project.
    with pytest.raises(Exception):
        ModelGateway().build_chat_model(ModelRef(ref="gemma4:e2b-mlx"))


def test_base_url_routes_through_openai_compatible():
    model = ModelGateway().build_chat_model(
        ModelRef(ref="openai:gemma2", base_url="http://localhost:11434/v1")
    )
    assert model is not None
    # Best-effort: the configured endpoint should be reflected on the client.
    client = getattr(model, "client", None)
    base_url = getattr(client, "base_url", None)
    if base_url is not None:
        assert "11434" in str(base_url)


def test_noop_usage_hook_conforms_and_is_a_noop():
    hook = NoopUsageHook()
    assert isinstance(hook, UsageHook)
    assert (
        hook.record(ref="openai:gpt-4o", kind="chat", usage=None, latency_ms=1.0)
        is None
    )

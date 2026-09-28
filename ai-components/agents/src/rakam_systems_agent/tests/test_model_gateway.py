"""Tests for the AI gateway chat factory (ModelGateway.build_chat_model)."""
import pytest

from rakam_systems_core.config_schema import ModelRef
from rakam_systems_agent.components.model_gateway import (
    ModelGateway,
    NoopUsageHook,
    UsageHook,
)


@pytest.fixture(autouse=True)
def _dummy_provider_keys(monkeypatch):
    # infer_model builds a client eagerly; give it a key so construction is hermetic.
    # MistralProvider raises UserError outright without one, so the mistral tests
    # below would otherwise fail for the wrong reason.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


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


# ==================== settings and client passthrough ====================


def test_model_without_settings_carries_none():
    # The pre-passthrough behaviour, pinned: a plain ref yields a model with no
    # settings of its own, so existing callers are untouched.
    model = ModelGateway().build_chat_model(ModelRef(ref="openai:gpt-4o"))
    assert model.settings is None


def test_settings_argument_reaches_the_model():
    from pydantic_ai.models.openai import OpenAIChatModelSettings

    model = ModelGateway().build_chat_model(
        ModelRef(ref="openai:gpt-4o"),
        settings=OpenAIChatModelSettings(
            temperature=0, seed=1234, max_tokens=512, extra_body={"store": False}
        ),
    )
    assert model.settings["temperature"] == 0
    assert model.settings["seed"] == 1234
    assert model.settings["max_tokens"] == 512
    # The zero-data-retention opt-out: a consumer under a retention obligation
    # cannot use the gateway at all unless this survives.
    assert model.settings["extra_body"] == {"store": False}


def test_settings_declared_on_the_ref_reach_the_model():
    # ModelRef allows extras, so config files can carry settings declaratively
    # without a schema change.
    model = ModelGateway().build_chat_model(
        ModelRef(
            ref="openai:gpt-4o",
            settings={"temperature": 0, "extra_body": {"store": False}},
        )
    )
    assert model.settings["temperature"] == 0
    assert model.settings["extra_body"] == {"store": False}


def test_settings_argument_overrides_the_ref_per_key():
    model = ModelGateway().build_chat_model(
        ModelRef(
            ref="openai:gpt-4o",
            settings={"temperature": 0.7, "extra_body": {"store": False}},
        ),
        settings={"temperature": 0, "seed": 7},
    )
    assert model.settings["temperature"] == 0
    assert model.settings["seed"] == 7
    # A key only the ref declares is kept, not clobbered by the override.
    assert model.settings["extra_body"] == {"store": False}


def test_settings_reach_the_model_on_the_base_url_path():
    model = ModelGateway().build_chat_model(
        ModelRef(ref="openai:gemma2", base_url="http://localhost:11434/v1"),
        settings={"temperature": 0},
    )
    assert model.settings["temperature"] == 0


def test_custom_openai_client_is_the_one_used():
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="test-key", max_retries=7)
    model = ModelGateway().build_chat_model(
        ModelRef(ref="openai:gpt-4o"), openai_client=client
    )
    assert model.client is client
    assert model.client.max_retries == 7


def test_custom_openai_client_reaches_a_non_openai_provider():
    # The client is threaded through pydantic-ai's own provider_factory seam, so
    # it works for any provider that accepts one -- here the Azure ZDR route,
    # which the gateway never names.
    from openai import AsyncAzureOpenAI

    client = AsyncAzureOpenAI(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",
        api_version="2024-06-01",
        max_retries=3,
    )
    model = ModelGateway().build_chat_model(
        ModelRef(ref="azure:gpt-4.1-mini"), openai_client=client
    )
    assert model.client is client


def test_custom_http_client_reaches_the_provider():
    import httpx

    http_client = httpx.AsyncClient()
    model = ModelGateway().build_chat_model(
        ModelRef(ref="openai:gpt-4o"), http_client=http_client
    )
    # `_client` is the openai SDK's own name for the httpx transport it wraps.
    assert model.client._client is http_client


def test_settings_and_custom_client_survive_together():
    # The exact shape the ingestion engine needs: pinned sampling + the
    # retention opt-out + a client carrying retries and an instrumented
    # transport, on one model.
    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModelSettings

    http_client = httpx.AsyncClient()
    client = AsyncOpenAI(api_key="test-key", max_retries=4, http_client=http_client)
    model = ModelGateway().build_chat_model(
        ModelRef(ref="openai:gpt-4o"),
        settings=OpenAIChatModelSettings(
            temperature=0, seed=99, extra_body={"store": False}
        ),
        openai_client=client,
    )
    assert model.client is client
    assert model.client.max_retries == 4
    assert model.client._client is http_client
    assert model.settings["temperature"] == 0
    assert model.settings["seed"] == 99
    assert model.settings["extra_body"] == {"store": False}


def test_settings_that_cannot_be_attached_raise_instead_of_being_dropped():
    # Guards the one private thing the gateway relies on: pydantic-ai's
    # Model.settings reflecting _settings. If a future version stopped doing so,
    # a retention opt-out would silently never reach the wire.
    from rakam_systems_agent.components.model_gateway.gateway import _attach_settings

    class _IgnoresSettings:
        settings = None

    with pytest.raises(RuntimeError, match="Model.settings"):
        _attach_settings(_IgnoresSettings(), {"extra_body": {"store": False}})


def test_base_url_and_openai_client_together_are_rejected():
    from openai import AsyncOpenAI

    with pytest.raises(ValueError, match="base_url"):
        ModelGateway().build_chat_model(
            ModelRef(ref="openai:gemma2", base_url="http://localhost:11434/v1"),
            openai_client=AsyncOpenAI(api_key="test-key"),
        )


# ==================== mistral ====================
#
# `mistral:` chat needs no code in this gateway -- infer_model reaches pydantic-ai's
# own MistralModel, which is the whole point of having no allow-list. These tests
# exist because that made it accidental: nothing pinned it, nothing documented it,
# and the embeddings half of the same gateway had no Mistral support at all.


def test_build_chat_model_resolves_mistral():
    model = ModelGateway().build_chat_model(ModelRef(ref="mistral:mistral-large-latest"))
    assert type(model).__name__ == "MistralModel"
    assert model.model_name == "mistral-large-latest"
    assert model.system == "mistral"


def test_settings_reach_a_mistral_model():
    # Also pins that _attach_settings' poke at pydantic-ai's private Model._settings
    # still lands for a second provider -- it raises rather than dropping silently.
    model = ModelGateway().build_chat_model(
        ModelRef(ref="mistral:mistral-small-latest"),
        settings={"temperature": 0, "max_tokens": 128},
    )
    assert model.settings["temperature"] == 0
    assert model.settings["max_tokens"] == 128


def test_custom_http_client_reaches_the_mistral_provider():
    import httpx

    http_client = httpx.AsyncClient()
    model = ModelGateway().build_chat_model(
        ModelRef(ref="mistral:mistral-large-latest"), http_client=http_client
    )
    # `async_client` is the mistralai SDK's own name for the httpx transport it wraps.
    assert model.client.sdk_configuration.async_client is http_client


def test_openai_client_is_rejected_for_a_mistral_ref():
    # The one place the deliberate absence of a provider-agnostic client seam bites.
    # MistralProvider takes `mistral_client`, not `openai_client`, and the gateway
    # threads the caller's argument through pydantic-ai's own provider_factory
    # without inspecting it -- so the SDK's own TypeError is what surfaces, naming
    # the argument. Documented behaviour, not an accident.
    from openai import AsyncOpenAI

    with pytest.raises(TypeError, match="openai_client"):
        ModelGateway().build_chat_model(
            ModelRef(ref="mistral:mistral-large-latest"),
            openai_client=AsyncOpenAI(api_key="test-key"),
        )


def test_mistral_ref_with_base_url_still_routes_through_openai_provider():
    # PINS A KNOWN ASYMMETRY, NOT A FEATURE. build_chat_model sends *any* ref that
    # carries a base_url through OpenAIProvider, so a mistral: ref plus a base_url
    # silently becomes an OpenAI-shaped client authenticating with OPENAI_API_KEY.
    # The embeddings half no longer behaves this way (build_embedder routes mistral:
    # to MistralEmbeddingModel first), so the two halves now disagree. Left alone
    # deliberately: fixing it means changing chat routing, which this change does not
    # cover. Tracked in docs/specs/ai-gateway.md's provider table.
    model = ModelGateway().build_chat_model(
        ModelRef(ref="mistral:mistral-large-latest", base_url="https://gw.internal")
    )
    assert type(model).__name__ == "OpenAIChatModel"
    assert model.system == "openai"


def test_mistral_without_a_key_surfaces_pydantic_ai_user_error(monkeypatch):
    # The gateway adds no credential handling of its own and swallows no error.
    from pydantic_ai.exceptions import UserError

    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(UserError, match="MISTRAL_API_KEY"):
        ModelGateway().build_chat_model(ModelRef(ref="mistral:mistral-large-latest"))

"""AI gateway model factory.

A thin factory over pydantic-ai: it turns a standardized :class:`ModelRef`
into a configured pydantic-ai model. It is a *factory*, not a call proxy --
the returned object is a plain pydantic-ai ``Model`` the caller uses directly
(``Agent(model=...)``), so pydantic-ai keeps owning streaming, tool-calls,
structured output and caching.

No provider allow-list: the ref is handed straight to pydantic-ai's
``infer_model`` so the full provider range stays reachable. An unknown
provider surfaces pydantic-ai's own error unchanged.

The factory also carries the two things a caller cannot express in a ref:
**model settings** (sampling and provider ``extra_body``) and an **already
built provider client**. Both are optional and default to ``None``, so a call
that passes neither behaves exactly as before.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from rakam_systems_core.config_schema import ModelRef
from rakam_systems_core.metering import NoopUsageHook, UsageHook

if TYPE_CHECKING:
    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings


class ModelGateway:
    """Builds pydantic-ai models from standardized ``ModelRef`` config."""

    def __init__(self, usage_hook: Optional[UsageHook] = None) -> None:
        # Stored for the (deferred) metering feature; not wired into chat yet.
        # Chat usage is a per-run value, forwarded at the run boundary later.
        self._usage_hook: UsageHook = usage_hook or NoopUsageHook()

    def build_chat_model(
        self,
        cfg: ModelRef,
        *,
        settings: Optional["ModelSettings"] = None,
        openai_client: Optional["AsyncOpenAI"] = None,
        http_client: Optional["httpx.AsyncClient"] = None,
    ) -> "Model":
        """Resolve ``cfg`` to a pydantic-ai chat model.

        When ``base_url`` is set, the ref is treated as an OpenAI-compatible
        endpoint (Ollama / local / OpenAI-compatible Azure) and routed through
        ``OpenAIProvider(base_url=...)``. Otherwise the ref is resolved by
        pydantic-ai's ``infer_model`` using the provider's standard env vars
        (e.g. ``AZURE_OPENAI_*`` for ``azure:``, ``OLLAMA_BASE_URL`` for
        ``ollama:``).

        Args:
            cfg: The model reference. ``cfg.settings`` -- an extra field, since
                ``ModelRef`` allows extras -- is the *declarative* place for
                settings that belong in a config file (``temperature``, ``seed``,
                ``max_tokens``, ``reasoning_effort``, ``extra_body``).
            settings: Model settings applied on top of ``cfg.settings``, merged
                per top-level key (so a nested value such as ``extra_body`` is
                replaced whole, not deep-merged). A pydantic-ai ``ModelSettings``
                (or a provider subclass such as ``OpenAIChatModelSettings``);
                both are ``TypedDict``s, so a plain dict works too. They become
                the model's *own* defaults, which pydantic-ai merges under
                per-run settings.
            openai_client: An already built ``AsyncOpenAI`` /
                ``AsyncAzureOpenAI``. Use it when the client itself carries
                configuration the ref cannot -- ``max_retries``, or an
                instrumented ``http_client``. Mutually exclusive with
                ``cfg.base_url``, which the client already encodes.
            http_client: An ``httpx.AsyncClient`` for the provider to build its
                own SDK client on. A provider whose constructor takes no such
                argument raises its own ``TypeError`` rather than silently
                dropping it -- ``bedrock:``, for instance.

        Returns:
            A pydantic-ai ``Model``, carrying ``settings`` as its defaults.

        Raises:
            ValueError: If both ``cfg.base_url`` and ``openai_client`` are given.
        """
        from pydantic_ai.models import infer_model

        merged = _merge_settings(getattr(cfg, "settings", None), settings)

        if openai_client is not None and cfg.base_url:
            raise ValueError(
                f"ModelRef(ref={cfg.ref!r}) sets base_url={cfg.base_url!r} and an "
                "openai_client was passed; the client already carries its own "
                "endpoint. Pass one or the other."
            )

        if cfg.base_url:
            from pydantic_ai.providers.openai import OpenAIProvider

            provider = OpenAIProvider(base_url=cfg.base_url, http_client=http_client)
            return _openai_chat_model(cfg.model_name, provider, merged)

        if openai_client is not None or http_client is not None:
            model = infer_model(
                cfg.ref,
                provider_factory=_client_provider_factory(openai_client, http_client),
            )
        else:
            model = infer_model(cfg.ref)

        return _attach_settings(model, merged)


def _merge_settings(
    base: Optional["ModelSettings"], override: Optional["ModelSettings"]
) -> Optional["ModelSettings"]:
    """Merge two settings mappings per key, ``override`` winning.

    Either side may be ``None``/empty, which means "nothing to say about
    settings" -- and if both are, the result is ``None`` rather than ``{}``, so
    the model is built exactly as it was before settings existed.
    """
    if not base:
        return override
    if not override:
        return base
    return {**base, **override}  # type: ignore[return-value]


def _attach_settings(model: "Model", settings: Optional["ModelSettings"]) -> "Model":
    """Make ``settings`` the model's own defaults.

    pydantic-ai models accept ``settings`` in their constructor, but
    ``infer_model`` -- the passthrough that keeps every provider reachable
    without an allow-list here -- does not forward it, and ``Model.settings`` is
    a read-only property over ``_settings``. So the backing attribute is set
    directly, and the result is verified: a silently dropped setting would mean
    a caller's ``extra_body={"store": False}`` retention opt-out never reaches
    the wire, which must fail loudly rather than quietly (a pydantic-ai rename
    is caught by tests/test_model_gateway.py).
    """
    if not settings:
        return model

    merged = _merge_settings(model.settings, settings)
    model._settings = merged  # type: ignore[attr-defined]
    if model.settings != merged:
        raise RuntimeError(
            "pydantic-ai's Model.settings no longer reflects _settings, so model "
            f"settings cannot be attached to {type(model).__name__}; "
            "pass them per-run via Agent(model_settings=...) until this is fixed"
        )
    return model


def _client_provider_factory(
    openai_client: Optional["AsyncOpenAI"], http_client: Optional["httpx.AsyncClient"]
):  # type: ignore[no-untyped-def]
    """Build the ``provider_factory`` ``infer_model`` uses to make the provider.

    Threading a caller's client through pydantic-ai's own factory seam is what
    keeps the no-allow-list guarantee: we decide only how the *provider* is
    constructed, never which model class a provider maps to. So the same
    argument serves ``openai:`` (``OpenAIProvider``) and the Azure ZDR route
    (``AzureProvider`` + ``AsyncAzureOpenAI``) without either being named here.
    """
    from pydantic_ai.providers import infer_provider_class

    kwargs: Dict[str, Any] = {}
    if openai_client is not None:
        kwargs["openai_client"] = openai_client
    if http_client is not None:
        kwargs["http_client"] = http_client

    def factory(provider_name: str):  # type: ignore[no-untyped-def]
        # A provider that takes no such argument raises its own TypeError naming
        # both, rather than silently dropping the caller's retry/instrumentation.
        return infer_provider_class(provider_name)(**kwargs)

    return factory


def _openai_chat_model(model_name: str, provider, settings=None):  # type: ignore[no-untyped-def]
    """Construct an OpenAI(-compatible) chat model across pydantic-ai versions.

    The class was renamed ``OpenAIModel`` -> ``OpenAIChatModel`` in newer
    pydantic-ai; try the current name first.
    """
    try:
        from pydantic_ai.models.openai import OpenAIChatModel as _Model
    except ImportError:  # pragma: no cover - older pydantic-ai
        from pydantic_ai.models.openai import OpenAIModel as _Model
    return _Model(model_name, provider=provider, settings=settings)

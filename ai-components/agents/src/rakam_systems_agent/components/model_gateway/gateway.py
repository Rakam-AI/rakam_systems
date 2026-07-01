"""AI gateway model factory.

A thin factory over pydantic-ai: it turns a standardized :class:`ModelRef`
into a configured pydantic-ai model. It is a *factory*, not a call proxy --
the returned object is a plain pydantic-ai ``Model`` the caller uses directly
(``Agent(model=...)``), so pydantic-ai keeps owning streaming, tool-calls,
structured output and caching.

No provider allow-list: the ref is handed straight to pydantic-ai's
``infer_model`` so the full provider range stays reachable. An unknown
provider surfaces pydantic-ai's own error unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rakam_systems_core.config_schema import ModelRef

from .metering import NoopUsageHook, UsageHook

if TYPE_CHECKING:
    from pydantic_ai.models import Model


class ModelGateway:
    """Builds pydantic-ai models from standardized ``ModelRef`` config."""

    def __init__(self, usage_hook: Optional[UsageHook] = None) -> None:
        # Stored for the (deferred) metering feature; not wired into chat yet.
        # Chat usage is a per-run value, forwarded at the run boundary later.
        self._usage_hook: UsageHook = usage_hook or NoopUsageHook()

    def build_chat_model(self, cfg: ModelRef) -> "Model":
        """Resolve ``cfg`` to a pydantic-ai chat model.

        When ``base_url`` is set, the ref is treated as an OpenAI-compatible
        endpoint (Ollama / local / OpenAI-compatible Azure) and routed through
        ``OpenAIProvider(base_url=...)``. Otherwise the ref is resolved by
        pydantic-ai's ``infer_model`` using the provider's standard env vars
        (e.g. ``AZURE_OPENAI_*`` for ``azure:``, ``OLLAMA_BASE_URL`` for
        ``ollama:``).
        """
        from pydantic_ai.models import infer_model

        if cfg.base_url:
            from pydantic_ai.providers.openai import OpenAIProvider

            provider = OpenAIProvider(base_url=cfg.base_url)
            return _openai_chat_model(cfg.model_name, provider)

        return infer_model(cfg.ref)


def _openai_chat_model(model_name: str, provider):  # type: ignore[no-untyped-def]
    """Construct an OpenAI(-compatible) chat model across pydantic-ai versions.

    The class was renamed ``OpenAIModel`` -> ``OpenAIChatModel`` in newer
    pydantic-ai; try the current name first.
    """
    try:
        from pydantic_ai.models.openai import OpenAIChatModel as _Model
    except ImportError:  # pragma: no cover - older pydantic-ai
        from pydantic_ai.models.openai import OpenAIModel as _Model
    return _Model(model_name, provider=provider)

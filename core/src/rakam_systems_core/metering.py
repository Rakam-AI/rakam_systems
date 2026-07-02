"""Usage-metering hook interface for the AI gateway.

The interface ships in OSS ``rakam_systems``; the real implementation (which
forwards token counts to the Rakam AI APIs collection) is proprietary and
injected privately at gateway construction. The payload is **counts/metadata
only -- never prompt/response content** -- so per-client data residency is
preserved. See the metering feature spec for the implementation.

It lives in ``rakam_systems_core`` (rather than the agent package) because
usage metering is cross-cutting: it spans both chat and embedding calls, so
both the agent-side ``ModelGateway`` and the vectorstore-side embedder can
depend on the same hook contract without depending on each other.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

Kind = Literal["chat", "embedding"]


@runtime_checkable
class UsageHook(Protocol):
    """Records post-call usage. Counts/metadata only, never content."""

    def record(self, *, ref: str, kind: Kind, usage: Any, latency_ms: float) -> None:
        ...


class NoopUsageHook:
    """Default hook: records nothing. Used when no metering is wired (OSS default)."""

    def record(self, *, ref: str, kind: Kind, usage: Any, latency_ms: float) -> None:
        return None

"""AI gateway: a thin pydantic-ai-backed model factory (chat + embeddings)."""
from rakam_systems_core.metering import NoopUsageHook, UsageHook

from .gateway import ModelGateway

# ``UsageHook``/``NoopUsageHook`` now live in ``rakam_systems_core.metering``
# (cross-cutting between chat and embedding); re-exported here so the gateway's
# public surface stays intact.
__all__ = ["ModelGateway", "UsageHook", "NoopUsageHook"]

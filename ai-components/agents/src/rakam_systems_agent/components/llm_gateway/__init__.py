"""LLM Gateway components for standardized multi-provider LLM interactions."""

from typing import TYPE_CHECKING

from rakam_systems_core.interfaces.llm_gateway import LLMGateway, LLMRequest, LLMResponse

from ..._optional import make_getattr as _make_getattr

if TYPE_CHECKING:  # import paths unchanged for type checkers
    from .gateway_factory import LLMGatewayFactory, get_llm_gateway
    from .mistral_gateway import MistralGateway
    from .openai_gateway import OpenAIGateway

# These carry the provider SDKs (openai / tiktoken / mistralai) from the
# optional [llm-providers] extra -- resolved on first access (see _optional).
__getattr__ = _make_getattr(
    __name__,
    ["OpenAIGateway", "MistralGateway", "LLMGatewayFactory", "get_llm_gateway"],
)

__all__ = [
    "LLMGateway",
    "LLMRequest",
    "LLMResponse",
    "OpenAIGateway",
    "MistralGateway",
    "LLMGatewayFactory",
    "get_llm_gateway",
]

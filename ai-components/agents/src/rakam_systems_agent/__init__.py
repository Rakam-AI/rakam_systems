"""
AI Agents Module

Provides flexible agent implementations with support for:
- Async/sync operations
- Tool integration
- Pydantic AI compatibility
- Streaming responses
- Multi-provider LLM gateway
"""

from typing import TYPE_CHECKING

from ._optional import make_getattr as _make_getattr
from .components import (
    BaseAgent, JSONChatHistory, LLMGateway,
    LLMRequest, LLMResponse, SQLChatHistory
)

if TYPE_CHECKING:  # import paths unchanged for type checkers
    from .components import (
        LLMGatewayFactory, MistralGateway, OpenAIGateway,
        PostgresChatHistory, get_llm_gateway
    )

# Names backed by an optional extra ([postgres] / [llm-providers]). They are
# resolved on first access, so importing this package -- and using the parts
# that need no provider SDK, e.g. ModelGateway -- works on a minimal install.
__getattr__ = _make_getattr(
    __name__,
    [
        "PostgresChatHistory",
        "OpenAIGateway",
        "MistralGateway",
        "LLMGatewayFactory",
        "get_llm_gateway",
    ],
)


__all__ = [
    "BaseAgent",
    "LLMGateway",
    "LLMRequest",
    "LLMResponse",
    "OpenAIGateway",
    "MistralGateway",
    "LLMGatewayFactory",
    "get_llm_gateway",
    "JSONChatHistory",
    "SQLChatHistory",
    "PostgresChatHistory"
]

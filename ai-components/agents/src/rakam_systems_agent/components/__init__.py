from typing import TYPE_CHECKING

from .._optional import make_getattr as _make_getattr
from .base_agent import BaseAgent
from .chat_history import JSONChatHistory, SQLChatHistory
from .llm_gateway import (
    LLMGateway,
    LLMRequest,
    LLMResponse,
)

if TYPE_CHECKING:  # import paths unchanged for type checkers
    from .chat_history import PostgresChatHistory
    from .llm_gateway import (
        LLMGatewayFactory,
        MistralGateway,
        OpenAIGateway,
        get_llm_gateway,
    )

# Backed by the optional [postgres] / [llm-providers] extras -- resolved on
# first access so the package imports without them (see _optional).
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
    "JSONChatHistory",
    "SQLChatHistory",
    "PostgresChatHistory",
    "LLMGateway",
    "LLMRequest",
    "LLMResponse",
    "OpenAIGateway",
    "MistralGateway",
    "LLMGatewayFactory",
    "get_llm_gateway",
]

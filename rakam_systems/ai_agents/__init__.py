"""
AI Agents Module

Provides flexible agent implementations with support for:
- Async/sync operations
- Tool integration
- Pydantic AI compatibility
- Streaming responses
- Multi-provider LLM gateway
"""

from .components import (
    BaseAgent,
    LLMGateway,
    LLMRequest,
    LLMResponse,
    OpenAIGateway,
    MistralGateway,
    LLMGatewayFactory,
    get_llm_gateway,
)

__version__ = "0.1.0"

__all__ = [
    "BaseAgent",
    "LLMGateway",
    "LLMRequest",
    "LLMResponse",
    "OpenAIGateway",
    "MistralGateway",
    "LLMGatewayFactory",
    "get_llm_gateway",
]


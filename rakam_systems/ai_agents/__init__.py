"""
AI Agents Module

Provides flexible agent implementations with support for:
- Async/sync operations
- Tool integration
- Pydantic AI compatibility
- Streaming responses
"""

from .components import BaseAgent, PydanticAIAgent

__version__ = "0.1.0"

__all__ = [
    "BaseAgent",
    "PydanticAIAgent",
]


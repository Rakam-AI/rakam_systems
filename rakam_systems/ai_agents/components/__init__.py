from .base_agent import BaseAgent

# Backward compatibility alias - PydanticAIAgent is now BaseAgent
PydanticAIAgent = BaseAgent

__all__ = [
    "BaseAgent",
    "PydanticAIAgent",
]


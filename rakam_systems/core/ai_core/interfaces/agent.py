from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterator, Optional, List, Type, TypeVar, Union, AsyncIterator
from ..base import BaseComponent

# Type variable for structured output
OutputT = TypeVar('OutputT')

class AgentInput:
    """Simple DTO for agent inputs (no external deps)."""
    def __init__(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.input_text = input_text
        self.context = context or {}

class AgentOutput(Generic[OutputT]):
    """Simple DTO for agent outputs.
    
    Supports both simple string outputs and structured outputs via output_type.
    When output_type is used, access the structured data via the `output` attribute.
    The `output_text` attribute provides a string representation for backward compatibility.
    """
    def __init__(
        self, 
        output_text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        output: Optional[OutputT] = None
    ) -> None:
        self.output_text = output_text
        self.metadata = metadata or {}
        self.output = output  # Structured output when output_type is used

class ModelSettings:
    """Settings for LLM model behavior."""
    def __init__(
        self, 
        parallel_tool_calls: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        self.parallel_tool_calls = parallel_tool_calls
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_settings = kwargs

class AgentComponent(BaseComponent, ABC):
    """Abstract agent interface with optional streaming and async support."""
    def __init__(
        self, 
        name: str, 
        config: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        deps_type: Optional[Type[Any]] = None,
        output_type: Optional[Type[Any]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self.stateful: bool = bool((config or {}).get("stateful", False))
        self.model = model or (config or {}).get("model", "openai:gpt-4")
        self.deps_type = deps_type
        self.output_type = output_type  # Pydantic model for structured output
        self.system_prompt = system_prompt or (config or {}).get("system_prompt", "")
        self.tools = tools or []

    @abstractmethod
    def run(self, input_data: Union[str, AgentInput], deps: Optional[Any] = None, model_settings: Optional[ModelSettings] = None) -> AgentOutput:
        """Synchronous run interface."""
        raise NotImplementedError

    async def arun(self, input_data: Union[str, AgentInput], deps: Optional[Any] = None, model_settings: Optional[ModelSettings] = None) -> AgentOutput:
        """Async run interface - override for true async support."""
        raise NotImplementedError

    def stream(self, input_data: Union[str, AgentInput], deps: Optional[Any] = None, model_settings: Optional[ModelSettings] = None) -> Iterator[str]:
        """Optional streaming interface: by default yields the final text once."""
        out = self.run(input_data, deps=deps, model_settings=model_settings)
        yield out.output_text

    async def astream(self, input_data: Union[str, AgentInput], deps: Optional[Any] = None, model_settings: Optional[ModelSettings] = None) -> AsyncIterator[str]:
        """Async streaming interface."""
        out = await self.arun(input_data, deps=deps, model_settings=model_settings)
        yield out.output_text

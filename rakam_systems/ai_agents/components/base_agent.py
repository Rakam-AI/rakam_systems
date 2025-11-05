from __future__ import annotations
from typing import Any, AsyncIterator, Iterator, List, Optional, Type, Union
from ai_core.interfaces.agent import AgentComponent, AgentInput, AgentOutput, ModelSettings
from ai_core.interfaces.tool import Tool

class BaseAgent(AgentComponent):
    """A convenient partial implementation of AgentComponent.
    Subclasses only need to implement `infer()` or `ainfer()`.
    """

    def __init__(
        self,
        name: str = "base_agent",
        config: Optional[dict] = None,
        model: Optional[str] = None,
        deps_type: Optional[Type[Any]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
    ) -> None:
        super().__init__(
            name=name,
            config=config,
            model=model,
            deps_type=deps_type,
            system_prompt=system_prompt,
            tools=tools,
        )

    def _normalize_input(self, input_data: Union[str, AgentInput]) -> AgentInput:
        """Convert string or AgentInput to AgentInput."""
        if isinstance(input_data, str):
            return AgentInput(input_text=input_data)
        return input_data

    def infer(
        self, 
        input_data: AgentInput, 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AgentOutput:
        """Override to implement non-streaming inference."""
        raise NotImplementedError

    async def ainfer(
        self, 
        input_data: AgentInput, 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AgentOutput:
        """Override to implement async non-streaming inference."""
        raise NotImplementedError

    def run(
        self, 
        input_data: Union[str, AgentInput], 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AgentOutput:
        normalized_input = self._normalize_input(input_data)
        return self.infer(normalized_input, deps=deps, model_settings=model_settings)

    async def arun(
        self, 
        input_data: Union[str, AgentInput], 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AgentOutput:
        normalized_input = self._normalize_input(input_data)
        return await self.ainfer(normalized_input, deps=deps, model_settings=model_settings)

    def stream(
        self, 
        input_data: Union[str, AgentInput], 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> Iterator[str]:
        # Default: produce a single chunk = final text
        normalized_input = self._normalize_input(input_data)
        out = self.infer(normalized_input, deps=deps, model_settings=model_settings)
        yield out.output_text

    async def astream(
        self, 
        input_data: Union[str, AgentInput], 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AsyncIterator[str]:
        # Default: produce a single chunk = final text
        normalized_input = self._normalize_input(input_data)
        out = await self.ainfer(normalized_input, deps=deps, model_settings=model_settings)
        yield out.output_text

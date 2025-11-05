from __future__ import annotations
from typing import Any, AsyncIterator, Iterator, List, Optional, Type, Union
from ai_core.interfaces.agent import AgentInput, AgentOutput, ModelSettings
from ai_core.interfaces.tool import Tool
from .base_agent import BaseAgent

try:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.settings import ModelSettings as PydanticModelSettings
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    PydanticAgent = None  # type: ignore
    PydanticModelSettings = None  # type: ignore


class PydanticAIAgent(BaseAgent):
    """Agent wrapper that uses Pydantic AI under the hood."""
    
    def __init__(
        self,
        name: str = "pydantic_agent",
        config: Optional[dict] = None,
        model: Optional[str] = None,
        deps_type: Optional[Type[Any]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
    ) -> None:
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "pydantic_ai is not installed. Please install it with: pip install pydantic_ai"
            )
        
        super().__init__(
            name=name,
            config=config,
            model=model,
            deps_type=deps_type,
            system_prompt=system_prompt,
            tools=tools,
        )
        
        # Initialize Pydantic AI agent
        self._pydantic_agent = PydanticAgent(
            model=self.model,
            deps_type=self.deps_type,
            system_prompt=self.system_prompt,
            tools=self._convert_tools_to_pydantic(tools or []),
        )
    
    def _convert_tools_to_pydantic(self, tools: List[Tool]) -> List[Any]:
        """Convert our Tool format to Pydantic AI Tool format."""
        if not PYDANTIC_AI_AVAILABLE:
            return []
        
        from pydantic_ai import Tool as PydanticTool
        
        pydantic_tools = []
        for tool in tools:
            pydantic_tool = PydanticTool.from_schema(
                function=tool.function,
                name=tool.name,
                description=tool.description,
                json_schema=tool.json_schema,
                takes_ctx=tool.takes_ctx,
            )
            pydantic_tools.append(pydantic_tool)
        
        return pydantic_tools
    
    def _convert_model_settings(self, model_settings: Optional[ModelSettings]) -> Optional[PydanticModelSettings]:
        """Convert our ModelSettings to Pydantic AI ModelSettings."""
        if model_settings is None or not PYDANTIC_AI_AVAILABLE:
            return None
        
        kwargs = {}
        
        # Only set parallel_tool_calls if agent has tools
        if self.tools:
            kwargs['parallel_tool_calls'] = model_settings.parallel_tool_calls
        
        if model_settings.temperature is not None:
            kwargs['temperature'] = model_settings.temperature
        
        if model_settings.max_tokens is not None:
            kwargs['max_tokens'] = model_settings.max_tokens
        
        kwargs.update(model_settings.extra_settings)
        
        return PydanticModelSettings(**kwargs)
    
    def infer(
        self, 
        input_data: AgentInput, 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AgentOutput:
        """Synchronous inference - not supported by Pydantic AI."""
        raise NotImplementedError(
            "PydanticAIAgent only supports async operations. Use ainfer() or arun() instead."
        )
    
    async def ainfer(
        self, 
        input_data: AgentInput, 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AgentOutput:
        """Async inference using Pydantic AI."""
        pydantic_settings = self._convert_model_settings(model_settings)
        
        # Run the Pydantic AI agent
        result = await self._pydantic_agent.run(
            input_data.input_text,
            deps=deps,
            model_settings=pydantic_settings,
        )
        
        # Convert result to our AgentOutput format
        output_text = result.output if hasattr(result, 'output') else str(result.data)
        metadata = {
            'usage': result.usage() if hasattr(result, 'usage') else None,
            'messages': result.messages() if hasattr(result, 'messages') else None,
        }
        
        return AgentOutput(output_text=output_text, metadata=metadata)
    
    def stream(
        self, 
        input_data: Union[str, AgentInput], 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> Iterator[str]:
        """Synchronous streaming - not supported by Pydantic AI."""
        raise NotImplementedError(
            "PydanticAIAgent only supports async operations. Use astream() instead."
        )
    
    async def astream(
        self, 
        input_data: Union[str, AgentInput], 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AsyncIterator[str]:
        """Async streaming using Pydantic AI."""
        normalized_input = self._normalize_input(input_data)
        pydantic_settings = self._convert_model_settings(model_settings)
        
        # Stream from the Pydantic AI agent
        async with self._pydantic_agent.run_stream(
            normalized_input.input_text,
            deps=deps,
            model_settings=pydantic_settings,
        ) as result:
            async for chunk in result.stream():
                yield chunk


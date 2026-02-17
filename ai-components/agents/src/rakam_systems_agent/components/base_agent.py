from __future__ import annotations
from typing import Any, AsyncIterator, Callable, Iterator, List, Optional, Type, Union
from rakam_systems_core.interfaces.agent import AgentComponent, AgentInput, AgentOutput, ModelSettings
from rakam_systems_core.interfaces.tool import ToolComponent
from rakam_systems_core.tracking import track_method, TrackingMixin

try:
    from rakam_systems_core.interfaces.tool_registry import ToolRegistry, ToolMode
    from rakam_systems_core.interfaces.tool_invoker import ToolInvoker
    TOOL_SYSTEM_AVAILABLE = True
except ImportError:
    ToolRegistry = None  # type: ignore
    ToolMode = None  # type: ignore
    ToolInvoker = None  # type: ignore
    TOOL_SYSTEM_AVAILABLE = False

try:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai import Tool as PydanticTool
    from pydantic_ai.settings import ModelSettings as PydanticModelSettings
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    PydanticAgent = None  # type: ignore
    PydanticTool = None  # type: ignore
    PydanticModelSettings = None  # type: ignore


# Type alias for dynamic system prompt functions
# Can be: () -> str, (ctx) -> str, async () -> str, or async (ctx) -> str
DynamicSystemPromptFunc = Callable[..., Union[str, Any]]


class BaseAgent(TrackingMixin, AgentComponent):
    """Base agent implementation using Pydantic AI.

    This is the core agent implementation in our system, powered by Pydantic AI.
    It supports both traditional tool lists and the new ToolRegistry/ToolInvoker system.
    When using a ToolRegistry, tools will be automatically loaded from the registry.

    Features:
    - Configuration-based initialization
    - Input/output tracking
    - Tool registry integration
    - Streaming support
    """

    def __init__(
        self,
        name: str = "base_agent",
        config: Optional[dict] = None,
        model: Optional[str] = None,
        deps_type: Optional[Type[Any]] = None,
        output_type: Optional[Type[Any]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolComponent]] = None,
        tool_registry: Optional[Any] = None,  # ToolRegistry
        tool_invoker: Optional[Any] = None,  # ToolInvoker
        enable_tracking: bool = False,
        tracking_output_dir: str = "./agent_tracking",
    ) -> None:
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "pydantic_ai is not installed. Please install it with: pip install pydantic_ai"
            )

        # Call super().__init__() which will handle both mixins properly
        super().__init__(
            name=name,
            config=config,
            model=model,
            deps_type=deps_type,
            output_type=output_type,
            system_prompt=system_prompt,
            tools=tools,
            enable_tracking=enable_tracking,
            tracking_output_dir=tracking_output_dir,
        )

        # Optional new tool system support
        self.tool_registry = tool_registry
        self.tool_invoker = tool_invoker

        # If registry is provided but no invoker, create one
        if tool_registry is not None and tool_invoker is None and TOOL_SYSTEM_AVAILABLE:
            self.tool_invoker = ToolInvoker(tool_registry)

        # Get tools from registry if provided, otherwise use tools list
        tools_to_use = self._get_tools_for_agent(tools, tool_registry)

        # Build kwargs for PydanticAgent, only including output_type if specified
        # (pydantic-ai defaults to str when not provided, but None causes issues)
        agent_kwargs = {
            "model": self.model,
            "deps_type": self.deps_type,
            "system_prompt": self.system_prompt,
            "tools": self._convert_tools_to_pydantic(tools_to_use),
        }
        if self.output_type is not None:
            agent_kwargs["output_type"] = self.output_type

        # Initialize Pydantic AI agent
        self._pydantic_agent = PydanticAgent(**agent_kwargs)

        # Store registered dynamic system prompt functions
        self._dynamic_system_prompts: List[DynamicSystemPromptFunc] = []

    def dynamic_system_prompt(
        self,
        func: Optional[DynamicSystemPromptFunc] = None
    ) -> Union[DynamicSystemPromptFunc, Callable[[DynamicSystemPromptFunc], DynamicSystemPromptFunc]]:
        """Register a dynamic system prompt function.

        This method can be used as a decorator to register functions that dynamically
        generate parts of the system prompt. The registered functions are passed directly
        to the underlying Pydantic AI agent.

        The decorated function can take an optional RunContext parameter and should return
        a string. It can be sync or async.

        Usage:
            ```python
            from datetime import date
            from pydantic_ai import RunContext

            agent = BaseAgent(
                name="my_agent",
                model="openai:gpt-4o",
                deps_type=str,
                system_prompt="Base system prompt."
            )

            @agent.dynamic_system_prompt
            def add_user_name(ctx: RunContext[str]) -> str:
                return f"The user's name is {ctx.deps}."

            @agent.dynamic_system_prompt
            def add_date() -> str:
                return f"The date is {date.today()}."

            # Now run the agent with deps
            result = await agent.arun("What is the date?", deps="Frank")
            ```

        Args:
            func: The function to register. If None, returns a decorator.

        Returns:
            The registered function (unchanged), or a decorator if func is None.
        """
        def decorator(f: DynamicSystemPromptFunc) -> DynamicSystemPromptFunc:
            # Register with the underlying Pydantic AI agent
            self._pydantic_agent.system_prompt(f)
            # Also keep track of it locally
            self._dynamic_system_prompts.append(f)
            return f

        if func is not None:
            # Used as @agent.dynamic_system_prompt without parentheses
            return decorator(func)
        else:
            # Used as @agent.dynamic_system_prompt() with parentheses
            return decorator

    def add_dynamic_system_prompt(self, func: DynamicSystemPromptFunc) -> DynamicSystemPromptFunc:
        """Add a dynamic system prompt function (non-decorator version).

        This is a convenience method for adding dynamic system prompts without
        using the decorator syntax.

        Usage:
            ```python
            def get_user_context(ctx: RunContext[str]) -> str:
                return f"User context: {ctx.deps}"

            agent.add_dynamic_system_prompt(get_user_context)
            ```

        Args:
            func: The function to register.

        Returns:
            The registered function (unchanged).
        """
        return self.dynamic_system_prompt(func)

    def _get_tools_for_agent(
        self,
        tools: Optional[List[ToolComponent]],
        tool_registry: Optional[Any]
    ) -> List[ToolComponent]:
        """Get tools from registry or use provided tools list."""
        if tools is not None:
            # Use explicitly provided tools
            return tools

        if tool_registry is not None:
            # Load direct tools from registry (MCP tools can't be used directly with agents)
            try:
                if TOOL_SYSTEM_AVAILABLE and ToolMode is not None:
                    direct_tools = tool_registry.get_tools_by_mode(
                        ToolMode.DIRECT)
                    result_tools = []
                    for metadata in direct_tools:
                        if metadata.tool_instance is not None:
                            result_tools.append(metadata.tool_instance)
                    return result_tools
            except (ImportError, AttributeError):
                pass

        # No tools available
        return []

    def _convert_tools_to_pydantic(self, tools: List[ToolComponent]) -> List[Any]:
        """Convert ToolComponent format to Pydantic AI Tool format."""
        if not PYDANTIC_AI_AVAILABLE:
            return []

        pydantic_tools = []
        for tool in tools:
            # ToolComponent now has all the attributes needed for Pydantic AI
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
        """Synchronous inference - not supported by Pydantic AI."""
        raise NotImplementedError(
            "BaseAgent only supports async operations. Use ainfer() or arun() instead."
        )

    @track_method()
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

        # Get the raw output from Pydantic AI
        raw_output = result.output if hasattr(
            result, 'output') else result.data

        # Convert result to our AgentOutput format
        # If output_type is used, raw_output will be the structured object
        if self.output_type is not None:
            output_text = str(raw_output)
            structured_output = raw_output
        else:
            output_text = raw_output if isinstance(
                raw_output, str) else str(raw_output)
            structured_output = None

        metadata = {
            'usage': result.usage() if hasattr(result, 'usage') else None,
            'messages': result.all_messages() if hasattr(result, 'all_messages') else None,
        }

        return AgentOutput(output_text=output_text, metadata=metadata, output=structured_output)

    def run(
        self,
        input_data: Union[str, AgentInput],
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AgentOutput:
        """Synchronous run - not supported by Pydantic AI."""
        raise NotImplementedError(
            "BaseAgent only supports async operations. Use arun() instead."
        )

    @track_method()
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
        """Synchronous streaming - not supported by Pydantic AI."""
        raise NotImplementedError(
            "BaseAgent only supports async operations. Use astream() instead."
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

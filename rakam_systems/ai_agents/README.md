# AI Agents Module

This module provides a flexible agent framework with support for both synchronous and asynchronous operations, tool integration, and Pydantic AI compatibility.

## Features

- **Async/Sync Support**: Full support for both synchronous and asynchronous agent operations
- **Tool Integration**: Easy tool definition and integration using the `Tool.from_schema` pattern
- **Model Settings**: Control model behavior including parallel tool calls, temperature, and max tokens
- **Pydantic AI Compatible**: Direct integration with Pydantic AI library
- **Streaming Support**: Both sync and async streaming interfaces

## Installation

Ensure you have the required dependencies:

```bash
pip install pydantic_ai
```

## Quick Start

### Using PydanticAIAgent

```python
import asyncio
from ai_agents.components import PydanticAIAgent
from ai_core.interfaces import ModelSettings, Tool

# Define a tool function
async def get_weather(city: str) -> dict:
    """Get weather information for a city"""
    # Your implementation here
    return {"city": city, "temperature": 72, "condition": "sunny"}

# Create an agent with tools
agent = PydanticAIAgent(
    name="weather_agent",
    model="openai:gpt-4o",
    system_prompt="You are a helpful weather assistant.",
    tools=[
        Tool.from_schema(
            function=get_weather,
            name='get_weather',
            description='Get weather information for a city',
            json_schema={
                'type': 'object',
                'properties': {
                    'city': {'type': 'string', 'description': 'The city name'},
                },
                'required': ['city'],
                'additionalProperties': False,
            },
            takes_ctx=False,
        ),
    ],
)

# Run the agent
async def main():
    result = await agent.arun(
        "What's the weather in San Francisco?",
        model_settings=ModelSettings(parallel_tool_calls=True),
    )
    print(result.output_text)

asyncio.run(main())
```

### Using BaseAgent for Custom Implementations

```python
from ai_agents.components import BaseAgent
from ai_core.interfaces import AgentInput, AgentOutput, ModelSettings

class MyCustomAgent(BaseAgent):
    async def ainfer(
        self, 
        input_data: AgentInput, 
        deps=None,
        model_settings=None
    ) -> AgentOutput:
        # Your custom implementation
        response = f"Processed: {input_data.input_text}"
        return AgentOutput(output_text=response)

# Use it
agent = MyCustomAgent(name="custom_agent")
result = await agent.arun("Hello, world!")
```

## Core Components

### AgentComponent

The base abstract class for all agents. Provides:
- `run()` / `arun()`: Execute the agent synchronously or asynchronously
- `stream()` / `astream()`: Stream responses
- Support for tools, model settings, and dependencies

### BaseAgent

A partial implementation of `AgentComponent` that handles input normalization and provides default implementations. Subclasses only need to implement `infer()` or `ainfer()`.

### PydanticAIAgent

A complete implementation that uses Pydantic AI under the hood. Provides:
- Direct integration with Pydantic AI's Agent
- Full support for parallel tool calls
- Automatic conversion between our interfaces and Pydantic AI's

### Tool

Wrapper for tool functions compatible with Pydantic AI's `Tool.from_schema` pattern:

```python
Tool.from_schema(
    function=my_function,
    name='my_function',
    description='What this function does',
    json_schema={
        'type': 'object',
        'properties': {...},
        'required': [...],
    },
    takes_ctx=False,
)
```

### ModelSettings

Configure model behavior:

```python
ModelSettings(
    parallel_tool_calls=True,  # Enable parallel tool execution
    temperature=0.7,           # Control randomness
    max_tokens=1000,           # Limit response length
)
```

## Advanced Usage

### Parallel vs Sequential Tool Calls

Control whether tools are called in parallel or sequentially:

```python
# Parallel (faster for independent tools)
result = await agent.arun(
    "Get weather for NYC and LA",
    model_settings=ModelSettings(parallel_tool_calls=True),
)

# Sequential (for dependent operations)
result = await agent.arun(
    "Get weather for NYC and LA",
    model_settings=ModelSettings(parallel_tool_calls=False),
)
```

### Using Dependencies

Pass context/dependencies to your agent:

```python
class Deps:
    def __init__(self, user_id: str):
        self.user_id = user_id

agent = PydanticAIAgent(
    deps_type=Deps,
    # ...
)

result = await agent.arun(
    "Process this",
    deps=Deps(user_id="123"),
)
```

### Streaming Responses

```python
async for chunk in agent.astream("Tell me a story"):
    print(chunk, end='', flush=True)
```

## API Reference

### AgentComponent

```python
class AgentComponent(BaseComponent):
    def __init__(
        self, 
        name: str,
        config: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        deps_type: Optional[Type[Any]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    )
    
    def run(
        self, 
        input_data: Union[str, AgentInput], 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AgentOutput
    
    async def arun(
        self, 
        input_data: Union[str, AgentInput], 
        deps: Optional[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ) -> AgentOutput
```

### Tool

```python
class Tool:
    @classmethod
    def from_schema(
        cls,
        function: Callable[..., Any],
        name: str,
        description: str,
        json_schema: Dict[str, Any],
        takes_ctx: bool = False,
    ) -> "Tool"
```

### ModelSettings

```python
class ModelSettings:
    def __init__(
        self, 
        parallel_tool_calls: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    )
```

## Examples

See `examples/pydantic_agent_example.py` for a complete example demonstrating:
- Multiple tool definitions
- Parallel vs sequential tool calls
- Performance comparisons
- Complex multi-tool workflows

## Architecture

```
ai_agents/
├── components/
│   ├── base_agent.py         # BaseAgent implementation
│   ├── pydantic_agent.py     # PydanticAIAgent wrapper
│   └── tools/                # Tool implementations
└── examples/
    └── pydantic_agent_example.py
```

## Best Practices

1. **Use async when possible**: Async operations are more efficient, especially with tools
2. **Enable parallel tool calls**: For independent operations, parallel execution is much faster
3. **Provide clear tool descriptions**: Better descriptions help the LLM use tools correctly
4. **Use type hints**: JSON schemas should match your function signatures
5. **Handle errors gracefully**: Tools should catch and return meaningful errors

## Migration Guide

If you're migrating from the old agent interface:

### Old Way
```python
agent = BaseAgent(name="agent")
result = agent.run(AgentInput(input_text="Hello"))
```

### New Way
```python
agent = PydanticAIAgent(name="agent")
result = await agent.arun("Hello")
```

## Troubleshooting

### ImportError: pydantic_ai not installed

Install Pydantic AI:
```bash
pip install pydantic_ai
```

### Tool not being called

Check:
1. Tool description is clear and relevant
2. JSON schema matches function signature
3. System prompt doesn't contradict tool usage

### Performance issues

- Enable `parallel_tool_calls=True` for independent operations
- Use async functions for I/O-bound operations
- Consider caching tool results when appropriate

## Contributing

When adding new agent types:
1. Inherit from `BaseAgent`
2. Implement `ainfer()` for async or `infer()` for sync
3. Add tests in `tests/`
4. Update this README with examples

## License

See main project LICENSE file.

---
title: Build agents
---

# Build agents

The agent package provides AI agent implementations powered by Pydantic AI. Install with `pip install rakam-systems-agent[all]` (requires core).

For a quick-start tutorial, see the [User Guide — Agents](../user-guide/agents.md).

## BaseAgent

```python
from rakam_systems_agent import BaseAgent
from rakam_systems_core.interfaces.agent import AgentInput, AgentOutput, ModelSettings

agent = BaseAgent(
    name="my_agent",
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
    tools=[my_tool],  # Optional tools
    output_type=MyOutputModel,  # Optional structured output
    enable_tracking=True  # Optional tracking
)

# Async inference (required for Pydantic AI)
result = await agent.arun("What is AI?")
print(result.output_text)

# With dependencies
result = await agent.arun("Hello", deps={"user_id": "123"})

# With model settings
settings = ModelSettings(temperature=0.5, max_tokens=1000)
result = await agent.arun("Explain quantum computing", model_settings=settings)

# Streaming
async for chunk in agent.astream("Tell me a story"):
    print(chunk, end="")
```

## Tools

Create tools with `ToolComponent.from_function`:

```python
from rakam_systems_core.interfaces.tool import ToolComponent

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny, 25°C"

weather_tool = ToolComponent.from_function(
    function=get_weather,
    name="get_weather",
    description="Get the current weather for a city",
    json_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
)

agent = BaseAgent(
    name="weather_assistant",
    model="openai:gpt-4o",
    system_prompt="You help users with weather information.",
    tools=[weather_tool]
)
```

## ToolRegistry

Central registry for managing tools across the system:

```python
from rakam_systems_core.interfaces.tool_registry import ToolRegistry, ToolMode

registry = ToolRegistry()

# Register a direct tool
registry.register_direct_tool(
    name="calculate",
    function=lambda x, y: x + y,
    description="Add two numbers",
    json_schema={...},
    category="math",
    tags=["arithmetic"]
)

# Register an MCP tool
registry.register_mcp_tool(
    name="search",
    mcp_server="search_server",
    mcp_tool_name="web_search",
    description="Search the web"
)

# Query tools
tools = registry.get_tools_by_category("math")
tools = registry.get_tools_by_tag("arithmetic")
tools = registry.get_tools_by_mode(ToolMode.DIRECT)
```

## Dynamic system prompts

Dynamic system prompts inject context at runtime based on current state, user information, or external data:

```python
from datetime import date, datetime
from pydantic_ai import RunContext

agent = BaseAgent(
    name="dynamic_agent",
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Method 1: Decorator syntax
@agent.dynamic_system_prompt
def add_date() -> str:
    """Add current date to system prompt."""
    return f"Today's date is {date.today().strftime('%B %d, %Y')}."

@agent.dynamic_system_prompt
def add_user_context(ctx: RunContext[dict]) -> str:
    """Add user-specific context from dependencies."""
    if ctx.deps and "user_name" in ctx.deps:
        return f"You are assisting {ctx.deps['user_name']}."
    return ""

# Method 2: Direct registration
def add_time_context() -> str:
    """Add current time to system prompt."""
    return f"Current time: {datetime.now().strftime('%H:%M:%S')}"

agent.add_dynamic_system_prompt(add_time_context)

# Method 3: Async dynamic prompts
@agent.dynamic_system_prompt
async def fetch_external_context(ctx: RunContext[dict]) -> str:
    """Fetch and add external context asynchronously."""
    import asyncio
    await asyncio.sleep(0.1)
    return "Additional context from external source."

# Usage with dependencies
result = await agent.arun(
    "What day is it?",
    deps={"user_name": "Alice", "user_id": "123"}
)
```

## Structured output

Pass an `output_type` to get typed responses:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

agent = BaseAgent(
    name="structured_agent",
    model="openai:gpt-4o",
    system_prompt="Answer questions with confidence scores.",
    output_type=Answer
)

result = await agent.arun("What is the capital of France?")
print(result.output.answer)       # "Paris"
print(result.output.confidence)   # 0.99
```

## Dependencies pattern

Pass runtime dependencies to agents and tools via `deps`:

```python
result = await agent.arun(
    "Look up order status",
    deps={"user_id": "123", "db_connection": db}
)
```

Dependencies are accessible in dynamic system prompts via `RunContext` and in tools that set `takes_ctx=True`.

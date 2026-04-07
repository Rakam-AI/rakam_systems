# Rakam Systems Agent

The agent package of Rakam Systems providing AI agent implementations powered by Pydantic AI.

## Overview

`rakam-systems-agent` provides flexible AI agents with tool integration, chat history, and LLM gateway abstractions. This package depends on `rakam-systems-core`.

## Features

- **Configuration-First Design**: Change agents without code changes — just update YAML files
- **Async/Sync Support**: Full support for both synchronous and asynchronous agent operations
- **Tool Integration**: Easy tool definition and integration using the `Tool.from_schema` pattern
- **Model Settings**: Control model behavior including parallel tool calls, temperature, and max tokens
- **Pydantic AI Powered**: Built on top of Pydantic AI library
- **Streaming Support**: Both sync and async streaming interfaces
- **Chat History**: Multiple backends (JSON, SQLite, PostgreSQL)
- **LLM Gateway**: Unified interface for OpenAI and Mistral AI

## Installation

```bash
pip install rakam-systems-agent
```

Available extras:

| Extra | What it adds |
|-------|-------------|
| `llm-providers` | `openai`, `mistralai`, `tiktoken` |
| `all` | Everything above |

```bash
pip install rakam-systems-agent[all]
```

## Quick Start

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_core.ai_core.interfaces import ModelSettings
from rakam_systems_core.ai_core.interfaces.tool import ToolComponent as Tool

async def get_weather(city: str) -> dict:
    """Get weather information for a city"""
    return {"city": city, "temperature": 72, "condition": "sunny"}

agent = BaseAgent(
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

async def main():
    result = await agent.arun(
        "What's the weather in San Francisco?",
        model_settings=ModelSettings(parallel_tool_calls=True),
    )
    print(result.output_text)

asyncio.run(main())
```

## Core Components

- **BaseAgent** — Primary agent implementation powered by Pydantic AI, with tool support and streaming
- **Tool** — Wrapper for tool functions using `Tool.from_schema` pattern
- **ModelSettings** — Configure parallel tool calls, temperature, max tokens
- **LLM Gateway** — Provider-agnostic interface for OpenAI and Mistral ([details](src/rakam_systems_agent/components/llm_gateway/README.md))
- **Chat History** — JSON, SQLite, and PostgreSQL backends
- **MCP Server** — Message-based component registry for agent tools ([details](src/rakam_systems_agent/server/README.md))

## Package Structure

```
rakam-systems-agent/
├── src/rakam_systems_agent/
│   ├── components/
│   │   ├── base_agent.py         # BaseAgent (Pydantic AI-powered)
│   │   ├── llm_gateway/          # LLM provider gateways
│   │   ├── chat_history/         # Chat history backends
│   │   ├── tools/                # Built-in tools
│   │   └── __init__.py           # Exports
│   └── server/                   # MCP server
└── pyproject.toml
```

## Documentation

For API reference, advanced usage (dependencies, streaming, YAML config), and troubleshooting, see the [official documentation](https://rakam-ai.github.io/rakam-systems-docs/).

## Contributing

When adding new agent types:

1. Inherit from `BaseAgent`
2. Implement `ainfer()` for async or `infer()` for sync
3. Add tests in `tests/`

## License

See main project LICENSE file.

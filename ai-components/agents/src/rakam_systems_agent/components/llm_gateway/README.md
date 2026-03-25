# LLM Gateway

A provider-agnostic LLM gateway for managing interactions with multiple LLM providers through a standardized interface.

## Overview

The LLM Gateway provides:

- **Multi-Provider Support**: OpenAI and Mistral (extensible to others)
- **Standardized Interface**: Consistent API across all providers
- **Configuration-Driven**: Select models via config files without code changes
- **Structured Outputs**: Type-safe responses using Pydantic schemas
- **Streaming Support**: Real-time response streaming
- **Token Counting**: Built-in token usage tracking

## Quick Start

```python
from rakam_systems_agent.components.llm_gateway import get_llm_gateway, LLMRequest

gateway = get_llm_gateway(model="openai:gpt-4o", temperature=0.7)

request = LLMRequest(
    system_prompt="You are a helpful assistant.",
    user_prompt="What is AI?",
    temperature=0.7,
)

response = gateway.generate(request)
print(response.content)
```

## Supported Providers

| Provider | Models | Env Variable |
|----------|--------|-------------|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` | `OPENAI_API_KEY` |
| Mistral | `mistral-large-latest`, `mistral-small-latest` | `MISTRAL_API_KEY` |

## Documentation

For full usage guide including structured outputs, streaming, factory patterns, and custom provider registration, see the [official documentation](https://rakam-ai.github.io/rakam-systems-docs/).

## License

See LICENSE file in the rakam_systems package root.

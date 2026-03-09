---
title: Use LLM gateways
---

# Use LLM gateways

LLM gateways provide a provider-agnostic interface for text generation, structured output, streaming, and token counting. Use them when you need direct LLM access outside of agents.

## OpenAI gateway

```python
from rakam_systems_agent import OpenAIGateway, LLMRequest

gateway = OpenAIGateway(
    model="gpt-4o",
    api_key="...",  # Or use OPENAI_API_KEY env var
    default_temperature=0.7
)

# Text generation
request = LLMRequest(
    system_prompt="You are a helpful assistant",
    user_prompt="What is AI?",
    temperature=0.7
)
response = gateway.generate(request)
print(response.content)

# Structured output
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

result = gateway.generate_structured(request, Answer)
print(result.answer, result.confidence)

# Streaming
for chunk in gateway.stream(request):
    print(chunk, end="")

# Token counting
token_count = gateway.count_tokens("Hello, world!")
```

## Mistral gateway

```python
from rakam_systems_agent import MistralGateway

gateway = MistralGateway(
    model="mistral-large-latest",
    api_key="..."  # Or use MISTRAL_API_KEY env var
)
```

The Mistral gateway exposes the same `generate`, `generate_structured`, `stream`, and `count_tokens` methods as the OpenAI gateway.

## Gateway factory

Create gateways dynamically by provider name:

```python
from rakam_systems_agent import LLMGatewayFactory, get_llm_gateway

# Using factory
gateway = LLMGatewayFactory.create(
    provider="openai",
    model="gpt-4o",
    api_key="..."
)

# Using convenience function
gateway = get_llm_gateway(provider="openai", model="gpt-4o")
```

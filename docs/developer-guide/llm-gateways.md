---
title: Use LLM gateways
---

# Use LLM gateways

LLM gateways provide a provider-agnostic interface for text generation, structured output, streaming, and token counting. Use them when you need direct LLM access outside of agents.

Two gateways coexist:

- **`ModelGateway`** — a thin factory that resolves a `provider:model` reference to a configured pydantic-ai `Model`. Use it when you drive the model through pydantic-ai (`Agent(model=...)`). Needs no provider SDK of its own.
- **`OpenAIGateway` / `MistralGateway`** (below) — raw provider-SDK gateways with their own `generate`/`stream`/`count_tokens` methods. They need the `llm-providers` extra.

## Model gateway

```python
from rakam_systems_core.config_schema import ModelRef
from rakam_systems_agent.components.model_gateway import ModelGateway

model = ModelGateway().build_chat_model(ModelRef(ref="openai:gpt-4o"))
```

There is no provider allow-list: the ref goes straight to pydantic-ai's `infer_model`, so every provider it supports is reachable, and an unknown provider surfaces pydantic-ai's own error. Set `base_url` for an OpenAI-compatible endpoint (Ollama, local, OpenAI-compatible Azure).

### Model settings

Sampling and provider-specific options travel as pydantic-ai model settings, either declared on the ref (they belong in a config file) or passed per call:

```python
from pydantic_ai.models.openai import OpenAIChatModelSettings

# Declarative: ModelRef allows extra fields, so config carries settings
model = ModelGateway().build_chat_model(
    ModelRef(
        ref="openai:gpt-4.1-mini",
        settings={"temperature": 0, "seed": 1234},
    )
)

# Per call, merged over the ref's settings key by key
model = ModelGateway().build_chat_model(
    ModelRef(ref="openai:gpt-4.1-mini"),
    settings=OpenAIChatModelSettings(
        temperature=0,
        seed=1234,
        max_tokens=512,
        extra_body={"store": False},   # OpenAI zero-data-retention opt-out
    ),
)
```

The settings become the model's *own* defaults, which pydantic-ai merges under anything passed per run (`Agent.run(model_settings=...)`).

Pin `temperature=0` and a recorded `seed` for extraction and classification work. Unpinned sampling is not a cosmetic default: it adds run-to-run variation that makes outputs non-reproducible and can swamp the difference you are actually trying to measure between two runs.

### Custom provider client

Pass an already built SDK client when it carries configuration the ref cannot — retry policy, or an instrumented HTTP transport:

```python
import httpx
from openai import AsyncOpenAI

client = AsyncOpenAI(max_retries=5, http_client=httpx.AsyncClient())
model = ModelGateway().build_chat_model(
    ModelRef(ref="openai:gpt-4o"),
    openai_client=client,
    settings={"temperature": 0, "extra_body": {"store": False}},
)
```

`openai_client` also serves the Azure route (`ref="azure:<deployment>"` with an `AsyncAzureOpenAI`). It is mutually exclusive with `ModelRef.base_url`, which the client already encodes. To let the provider build its own SDK client on your transport, pass `http_client=` instead.

## Optional extras

Provider SDKs are optional: `pip install rakam-systems-agent` installs neither `openai`/`mistralai`/`tiktoken` (extra `llm-providers`) nor `psycopg2` (extra `postgres`). Importing the package and using `BaseAgent`, `ModelGateway`, `JSONChatHistory` or `SQLChatHistory` needs no extra. `OpenAIGateway`, `MistralGateway`, `LLMGatewayFactory`, `get_llm_gateway` and `PostgresChatHistory` are resolved on first access and raise an `ImportError` naming the extra to install if it is missing.

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

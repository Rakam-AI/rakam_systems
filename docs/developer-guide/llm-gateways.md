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

There is no provider allow-list: the ref goes straight to pydantic-ai's `infer_model`, so every provider it supports is reachable, and an unknown provider surfaces pydantic-ai's own error. Mistral needs no extra of its own here — `MISTRAL_API_KEY` is all it takes:

```python
model = ModelGateway().build_chat_model(ModelRef(ref="mistral:mistral-large-latest"))
```

Set `base_url` for an OpenAI-compatible endpoint (Ollama, local, OpenAI-compatible Azure). Note that `base_url` routes *any* ref through the OpenAI-compatible path, so `mistral:<model>` with a `base_url` gives you an OpenAI-shaped client on `OPENAI_API_KEY`, not a Mistral one. To reach an OpenAI-compatible proxy, write the ref as `openai:<model>`.

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

The argument is threaded through pydantic-ai's own provider factory rather than interpreted here, so it only fits providers that accept an `openai_client`. On a `mistral:` ref it raises `TypeError: MistralProvider.__init__() got an unexpected keyword argument 'openai_client'` — `http_client=` works there instead.

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

The Mistral gateway exposes the same `generate`, `generate_structured`, `stream`, and `count_tokens` methods as the OpenAI gateway, plus three optional settings.

### Custom endpoint

```python
gateway = MistralGateway(model="mistral-large-latest", base_url="https://gw.internal")
```

Give the origin. mistralai appends `/v1/...` itself, so a trailing `/v1` is stripped for you and `https://gw.internal`, `https://gw.internal/` and `https://gw.internal/v1` all reach the same place — one config value works for both this gateway and `OpenAIGateway`. `LLMGatewayFactory.create_gateway_from_config` forwards `base_url` for Mistral too (before, it silently dropped it).

### Structured output

`generate_structured` asks for Mistral's native strict `json_schema` format by default and falls back to describing the schema in the system prompt if the model rejects it:

```python
gateway = MistralGateway(model="mistral-large-latest", structured_mode="auto")
```

- `"auto"` (default) — try strict mode; on the first rejection, downgrade *this gateway* to `"json_object"` and log a warning. One extra request per gateway, not per call.
- `"json_schema"` — strict mode only; a model that does not support it raises.
- `"json_object"` — the pre-existing behaviour, byte for byte. Use it to pin the old wire format.

### Token counting

`count_tokens` approximates at 4 characters per token unless you supply a counter:

```python
gateway = MistralGateway(model="mistral-large-latest", token_counter=my_counter)
```

Mistral publishes no tokenize endpoint, and `mistral-common` cannot resolve a `-latest` model name offline, so an exact *local* count has to come from you. **If you are counting a prompt you are about to send, don't** — the API reports the exact number on every response as `LLMResponse.usage.prompt_tokens`. Pass `token_counting="exact"` to make a missing or failing counter an error instead of a silent fall back to the approximation.

## Gateway factory

Create gateways dynamically by provider name:

```python
from rakam_systems_agent import LLMGatewayFactory, get_llm_gateway

# From a "provider:model" string
gateway = LLMGatewayFactory.create_gateway(
    model_string="mistral:mistral-large-latest",
    temperature=0.7,
    api_key="...",
)

# From a config dict (provider-specific keys such as base_url are forwarded)
gateway = LLMGatewayFactory.create_gateway_from_config({
    "provider": "mistral",
    "model": "mistral-large-latest",
    "temperature": 0.7,
    "base_url": "https://gw.internal",
})

# Using convenience function
gateway = get_llm_gateway(provider="openai", model="gpt-4o")
```

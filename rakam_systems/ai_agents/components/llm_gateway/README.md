# LLM Gateway System

A centralized, provider-agnostic LLM gateway for managing interactions with multiple LLM providers through a standardized interface.

## Overview

The LLM Gateway system provides:

- ✅ **Multi-Provider Support**: OpenAI and Mistral (extensible to others)
- ✅ **Standardized Interface**: Consistent API across all providers
- ✅ **Configuration-Driven**: Select models via config files without code changes
- ✅ **Structured Outputs**: Type-safe responses using Pydantic schemas
- ✅ **Streaming Support**: Real-time response streaming
- ✅ **Token Counting**: Built-in token usage tracking
- ✅ **Factory Pattern**: Easy provider routing and model selection

## Architecture

```
┌─────────────────────────────────────────────────────┐
│            LLMGatewayFactory                        │
│  (Provider routing & configuration)                 │
└───────────────┬────────────────────┬────────────────┘
                │                    │
    ┌───────────▼────────┐  ┌───────▼──────────┐
    │  OpenAIGateway     │  │  MistralGateway  │
    └───────────┬────────┘  └───────┬──────────┘
                │                    │
                └────────┬───────────┘
                         │
                ┌────────▼────────┐
                │   LLMGateway    │
                │  (Base Class)   │
                └─────────────────┘
```

## Quick Start

### Basic Usage

```python
from ai_agents.components.llm_gateway import get_llm_gateway, LLMRequest

# Create a gateway using model string
gateway = get_llm_gateway(model="openai:gpt-4o", temperature=0.7)

# Make a request
request = LLMRequest(
    system_prompt="You are a helpful assistant.",
    user_prompt="What is AI?",
    temperature=0.7,
)

response = gateway.generate(request)
print(response.content)
print(f"Tokens used: {response.usage}")
```

### Structured Output

```python
from pydantic import BaseModel, Field
from ai_agents.components.llm_gateway import OpenAIGateway, LLMRequest

class Book(BaseModel):
    title: str = Field(description="Book title")
    author: str = Field(description="Author name")
    year: int = Field(description="Publication year")

gateway = OpenAIGateway(model="gpt-4o")
request = LLMRequest(
    system_prompt="You are a librarian.",
    user_prompt="Tell me about '1984' by George Orwell.",
)

book = gateway.generate_structured(request, Book)
print(f"{book.title} by {book.author} ({book.year})")
```

### Streaming Responses

```python
from ai_agents.components.llm_gateway import get_llm_gateway, LLMRequest

gateway = get_llm_gateway(model="openai:gpt-4o")
request = LLMRequest(
    user_prompt="Write a short story about AI.",
    temperature=0.8,
)

for chunk in gateway.stream(request):
    print(chunk, end="", flush=True)
```

## Configuration-Driven Usage

### YAML Configuration

```yaml
# config.yaml
llm_gateways:
  default:
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.7
    max_tokens: 2000
  
  creative:
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.9
    max_tokens: 3000
  
  analytical:
    provider: "mistral"
    model: "mistral-large-latest"
    temperature: 0.3
```

### Load from Configuration

```python
from ai_agents.components.llm_gateway import LLMGatewayFactory

# Load config (pseudo-code)
config = load_yaml_config("config.yaml")

# Create gateways from config
default_gateway = LLMGatewayFactory.create_gateway_from_config(
    config["llm_gateways"]["default"]
)

creative_gateway = LLMGatewayFactory.create_gateway_from_config(
    config["llm_gateways"]["creative"]
)
```

## Factory Patterns

### Pattern 1: Model String

```python
from ai_agents.components.llm_gateway import LLMGatewayFactory

# Provider:model format
gateway = LLMGatewayFactory.create_gateway("openai:gpt-4o")

# Auto-detect provider (defaults to OpenAI)
gateway = LLMGatewayFactory.create_gateway("gpt-4o")

# Mistral
gateway = LLMGatewayFactory.create_gateway("mistral:mistral-large-latest")
```

### Pattern 2: Configuration Dictionary

```python
config = {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "base_url": "https://api.openai.com/v1",  # optional
}

gateway = LLMGatewayFactory.create_gateway_from_config(config)
```

### Pattern 3: Environment Variables

```python
# Set environment variables
# DEFAULT_LLM_MODEL=openai:gpt-4o
# DEFAULT_LLM_TEMPERATURE=0.7

gateway = LLMGatewayFactory.get_default_gateway()
```

### Pattern 4: Direct Instantiation

```python
from ai_agents.components.llm_gateway import OpenAIGateway, MistralGateway

openai_gateway = OpenAIGateway(
    model="gpt-4o",
    default_temperature=0.7,
    api_key="your-key",  # or use OPENAI_API_KEY env var
)

mistral_gateway = MistralGateway(
    model="mistral-large-latest",
    default_temperature=0.7,
    api_key="your-key",  # or use MISTRAL_API_KEY env var
)
```

## Provider Details

### OpenAI

**Supported Models:**
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-4`
- `gpt-3.5-turbo`

**Features:**
- Native structured output support via `response_format`
- Accurate token counting with `tiktoken`
- Streaming support
- Custom base URL support

**Configuration:**
```python
gateway = OpenAIGateway(
    model="gpt-4o",
    default_temperature=0.7,
    api_key="sk-...",  # or OPENAI_API_KEY env var
    base_url="https://api.openai.com/v1",  # optional
    organization="org-...",  # optional
)
```

### Mistral

**Supported Models:**
- `mistral-large-latest`
- `mistral-medium-latest`
- `mistral-small-latest`
- `open-mistral-7b`
- `open-mixtral-8x7b`

**Features:**
- JSON mode for structured outputs
- Approximate token counting
- Streaming support

**Configuration:**
```python
gateway = MistralGateway(
    model="mistral-large-latest",
    default_temperature=0.7,
    api_key="...",  # or MISTRAL_API_KEY env var
)
```

## Standardized Request/Response

### LLMRequest

```python
class LLMRequest(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[str] = None
    json_schema: Optional[Type[BaseModel]] = None
    extra_params: Dict[str, Any] = {}
```

### LLMResponse

```python
class LLMResponse(BaseModel):
    content: str
    parsed_content: Optional[Any] = None
    usage: Optional[Dict[str, Any]] = None  # token counts
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = {}
```

## Advanced Features

### Token Counting

```python
gateway = get_llm_gateway(model="openai:gpt-4o")

text = "Hello, world!"
token_count = gateway.count_tokens(text)
print(f"Tokens: {token_count}")
```

### Custom Provider Registration

```python
from ai_agents.components.llm_gateway import LLMGatewayFactory, LLMGateway

class CustomGateway(LLMGateway):
    # Implement required methods
    pass

LLMGatewayFactory.register_provider(
    provider_name="custom",
    gateway_class=CustomGateway,
    default_model="custom-model-v1",
)

# Now you can use it
gateway = LLMGatewayFactory.create_gateway("custom:custom-model-v1")
```

### List Available Providers

```python
providers = LLMGatewayFactory.list_providers()
print(f"Available providers: {providers}")
# Output: ['openai', 'mistral']

for provider in providers:
    default_model = LLMGatewayFactory.get_default_model(provider)
    print(f"{provider}: {default_model}")
```

## Error Handling

```python
from ai_agents.components.llm_gateway import LLMGatewayFactory

try:
    gateway = LLMGatewayFactory.create_gateway("invalid:model")
except ValueError as e:
    print(f"Error: {e}")
    # Error: Unknown provider 'invalid'

try:
    gateway = OpenAIGateway(model="gpt-4o")
    # Missing OPENAI_API_KEY
except ValueError as e:
    print(f"Error: {e}")
    # Error: OpenAI API key must be provided
```

## Environment Variables

The gateway system respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI |
| `MISTRAL_API_KEY` | Mistral API key | Required for Mistral |
| `DEFAULT_LLM_MODEL` | Default model string | `openai:gpt-4o` |
| `DEFAULT_LLM_PROVIDER` | Default provider | `openai` |
| `DEFAULT_LLM_TEMPERATURE` | Default temperature | `0.7` |

## Best Practices

### 1. Use Configuration Files

Store gateway configurations in YAML files for easy management:

```yaml
llm_gateways:
  production:
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.7
  
  development:
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.7
```

### 2. Environment-Based API Keys

Never hardcode API keys. Use environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export MISTRAL_API_KEY="..."
```

### 3. Handle Provider Differences

While the interface is standardized, be aware of provider-specific behaviors:

```python
# OpenAI has native structured output
openai_result = openai_gateway.generate_structured(request, Schema)

# Mistral uses JSON mode (may require schema in prompt)
mistral_result = mistral_gateway.generate_structured(request, Schema)
```

### 4. Use Appropriate Models

Choose models based on your use case:

- **Fast, cost-effective**: `gpt-4o-mini`, `mistral-small-latest`
- **High quality**: `gpt-4o`, `mistral-large-latest`
- **Creative tasks**: Higher temperature (0.8-0.9)
- **Analytical tasks**: Lower temperature (0.2-0.4)

### 5. Monitor Token Usage

Always check token usage to manage costs:

```python
response = gateway.generate(request)
if response.usage:
    print(f"Tokens: {response.usage['total_tokens']}")
    print(f"Cost estimate: ${response.usage['total_tokens'] * 0.00001}")
```

## Integration with Agents

The LLM Gateway can be used alongside the BaseAgent system:

```python
from ai_agents import BaseAgent
from ai_agents.components.llm_gateway import get_llm_gateway

# Create a gateway for custom LLM calls
gateway = get_llm_gateway(model="openai:gpt-4o")

# Use BaseAgent for agent framework (uses Pydantic AI internally)
agent = BaseAgent(
    name="my_agent",
    model="openai:gpt-4o",  # Pydantic AI model string
    system_prompt="You are a helpful assistant.",
)

# Gateway for direct LLM calls, agent for tool-using conversations
```

## Examples

See the complete examples in:
- `examples/llm_gateway_example.py` - Comprehensive usage examples
- `examples/configs/llm_gateway_config.yaml` - Configuration examples

Run examples:
```bash
cd rakam_systems
python -m examples.llm_gateway_example
```

## API Reference

### LLMGateway (Base Class)

**Methods:**
- `generate(request: LLMRequest) -> LLMResponse`
- `generate_structured(request: LLMRequest, schema: Type[T]) -> T`
- `stream(request: LLMRequest) -> Iterator[str]`
- `count_tokens(text: str, model: Optional[str]) -> int`

### LLMGatewayFactory

**Methods:**
- `create_gateway(model_string: str, **kwargs) -> LLMGateway`
- `create_gateway_from_config(config: Dict) -> LLMGateway`
- `get_default_gateway() -> LLMGateway`
- `register_provider(name: str, gateway_class: Type, default_model: str)`
- `list_providers() -> List[str]`
- `get_default_model(provider: str) -> Optional[str]`

### Convenience Functions

- `get_llm_gateway(model: str, **kwargs) -> LLMGateway`

## Dependencies

- `openai>=1.0.0` - OpenAI Python SDK
- `mistralai>=0.1.0` - Mistral Python SDK
- `tiktoken>=0.5.0` - Token counting for OpenAI
- `pydantic>=2.0.0` - Data validation and structured outputs

## License

See LICENSE file in the rakam_systems package root.

## Contributing

To add a new provider:

1. Create a new gateway class inheriting from `LLMGateway`
2. Implement all required abstract methods
3. Register with `LLMGatewayFactory.register_provider()`
4. Add tests and documentation

Example:

```python
class NewProviderGateway(LLMGateway):
    def generate(self, request: LLMRequest) -> LLMResponse:
        # Implementation
        pass
    
    def generate_structured(self, request: LLMRequest, schema: Type[T]) -> T:
        # Implementation
        pass
    
    def count_tokens(self, text: str, model: Optional[str]) -> int:
        # Implementation
        pass
```

## Support

For issues, questions, or contributions, please refer to the main rakam_systems repository.


# LLM Gateway - Quick Start Guide

## Installation

Add required dependencies (already in requirements.txt):
```bash
pip install openai>=1.37.0 mistralai>=1.0.0 tiktoken>=0.5.0
```

## Setup API Keys

```bash
export OPENAI_API_KEY="sk-..."
export MISTRAL_API_KEY="..."
```

## 5-Minute Quick Start

### 1. Basic Text Generation

```python
from ai_agents.components.llm_gateway import get_llm_gateway, LLMRequest

# Create gateway
gateway = get_llm_gateway(model="openai:gpt-4o-mini")

# Make request
request = LLMRequest(
    system_prompt="You are a helpful assistant",
    user_prompt="Explain quantum computing in one sentence"
)

response = gateway.generate(request)
print(response.content)
```

### 2. Switch Providers (No Code Change!)

**Via Configuration:**
```yaml
# config.yaml
llm_gateways:
  main:
    provider: "mistral"  # Changed from "openai"
    model: "mistral-large-latest"
    temperature: 0.7
```

**Via Environment:**
```bash
export DEFAULT_LLM_MODEL="mistral:mistral-large-latest"
```

**Via Code:**
```python
# Just change the model string
gateway = get_llm_gateway(model="mistral:mistral-large-latest")
# Everything else stays the same!
```

### 3. Structured Output

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    category: str = Field(description="Product category")

gateway = get_llm_gateway("openai:gpt-4o-mini")
request = LLMRequest(
    user_prompt="Create a product listing for a laptop"
)

product = gateway.generate_structured(request, Product)
print(f"{product.name}: ${product.price}")
```

### 4. Streaming

```python
gateway = get_llm_gateway("openai:gpt-4o-mini")
request = LLMRequest(user_prompt="Write a haiku about AI")

for chunk in gateway.stream(request):
    print(chunk, end="", flush=True)
```

## Common Patterns

### Pattern: Configuration File

```yaml
# app_config.yaml
llm_gateways:
  fast:
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.7
  
  powerful:
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.5
  
  alternative:
    provider: "mistral"
    model: "mistral-large-latest"
    temperature: 0.7
```

```python
import yaml
from ai_agents.components.llm_gateway import LLMGatewayFactory

with open("app_config.yaml") as f:
    config = yaml.safe_load(f)

# Use different gateways for different needs
fast_gateway = LLMGatewayFactory.create_gateway_from_config(
    config["llm_gateways"]["fast"]
)

powerful_gateway = LLMGatewayFactory.create_gateway_from_config(
    config["llm_gateways"]["powerful"]
)
```

### Pattern: Dynamic Provider Selection

```python
def get_gateway_for_task(task_type: str):
    """Select gateway based on task requirements."""
    configs = {
        "summarization": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3},
        "creative": {"provider": "openai", "model": "gpt-4o", "temperature": 0.9},
        "analysis": {"provider": "mistral", "model": "mistral-large-latest", "temperature": 0.2},
    }
    
    return LLMGatewayFactory.create_gateway_from_config(configs[task_type])

# Use appropriate gateway for each task
summary_gateway = get_gateway_for_task("summarization")
creative_gateway = get_gateway_for_task("creative")
```

### Pattern: Token Management

```python
gateway = get_llm_gateway("openai:gpt-4o")

# Count tokens before sending
prompt_text = "Your prompt here"
tokens = gateway.count_tokens(prompt_text)
print(f"This will use approximately {tokens} tokens")

# Check usage after response
response = gateway.generate(request)
if response.usage:
    print(f"Actual tokens used: {response.usage['total_tokens']}")
    cost = response.usage['total_tokens'] * 0.00001  # Approximate cost
    print(f"Estimated cost: ${cost:.4f}")
```

## Supported Providers & Models

### OpenAI
- `gpt-4o` - Most capable
- `gpt-4o-mini` - Fast and cost-effective
- `gpt-4-turbo` - Previous generation
- `gpt-4` - Original GPT-4
- `gpt-3.5-turbo` - Legacy

### Mistral
- `mistral-large-latest` - Most capable
- `mistral-medium-latest` - Balanced
- `mistral-small-latest` - Fast and economical

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Quick Q&A | `gpt-4o-mini` | Fast, cheap, good enough |
| Complex reasoning | `gpt-4o` | Best quality |
| Creative writing | `gpt-4o` @ temp 0.9 | Better creativity |
| Data extraction | `gpt-4o-mini` @ temp 0.2 | Consistent output |
| Cost-sensitive | `mistral-small-latest` | Lower cost |
| Non-OpenAI option | `mistral-large-latest` | Alternative provider |

## Error Handling

```python
from ai_agents.components.llm_gateway import get_llm_gateway, LLMRequest

try:
    gateway = get_llm_gateway("openai:gpt-4o")
    request = LLMRequest(user_prompt="Hello")
    response = gateway.generate(request)
    print(response.content)
    
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `MISTRAL_API_KEY` | Mistral API key | `...` |
| `DEFAULT_LLM_MODEL` | Default model | `openai:gpt-4o` |
| `DEFAULT_LLM_PROVIDER` | Default provider | `openai` |
| `DEFAULT_LLM_TEMPERATURE` | Default temperature | `0.7` |

## Complete Example: Multi-Task Agent

```python
from ai_agents.components.llm_gateway import get_llm_gateway, LLMRequest
from pydantic import BaseModel, Field
from typing import List

class TaskResult(BaseModel):
    task: str
    result: str
    tokens_used: int

class Agent:
    def __init__(self):
        # Different gateways for different tasks
        self.creative = get_llm_gateway("openai:gpt-4o", temperature=0.9)
        self.analytical = get_llm_gateway("mistral:mistral-large-latest", temperature=0.3)
        self.fast = get_llm_gateway("openai:gpt-4o-mini", temperature=0.7)
    
    def creative_task(self, prompt: str) -> TaskResult:
        request = LLMRequest(user_prompt=prompt)
        response = self.creative.generate(request)
        return TaskResult(
            task="creative",
            result=response.content,
            tokens_used=response.usage.get("total_tokens", 0) if response.usage else 0
        )
    
    def analytical_task(self, prompt: str) -> TaskResult:
        request = LLMRequest(user_prompt=prompt)
        response = self.analytical.generate(request)
        return TaskResult(
            task="analytical",
            result=response.content,
            tokens_used=response.usage.get("total_tokens", 0) if response.usage else 0
        )
    
    def quick_task(self, prompt: str) -> TaskResult:
        request = LLMRequest(user_prompt=prompt)
        response = self.fast.generate(request)
        return TaskResult(
            task="quick",
            result=response.content,
            tokens_used=response.usage.get("total_tokens", 0) if response.usage else 0
        )

# Usage
agent = Agent()

# Creative writing
story = agent.creative_task("Write a short sci-fi story opening")
print(f"Story: {story.result} (Tokens: {story.tokens_used})")

# Data analysis
analysis = agent.analytical_task("Analyze the sentiment of: 'I love this product!'")
print(f"Analysis: {analysis.result} (Tokens: {analysis.tokens_used})")

# Quick response
quick = agent.quick_task("What is 2+2?")
print(f"Quick: {quick.result} (Tokens: {quick.tokens_used})")
```

## Next Steps

1. **Read Full Documentation**: `ai_agents/components/llm_gateway/README.md`
2. **Run Examples**: `python -m examples.llm_gateway_example`
3. **Review Configuration**: `examples/configs/llm_gateway_config.yaml`
4. **Integration Guide**: `LLM_GATEWAY_IMPLEMENTATION.md`

## Tips & Tricks

1. **Use Mini for Development**: Save costs by using `gpt-4o-mini` during development
2. **Cache Common Requests**: Implement caching layer for repeated queries
3. **Monitor Costs**: Always track token usage
4. **Test Provider Switching**: Ensure your code works with both providers
5. **Set Timeouts**: Add timeout handling for production use

## Troubleshooting

**Problem**: `ValueError: API key must be provided`  
**Solution**: Set environment variable: `export OPENAI_API_KEY="sk-..."`

**Problem**: `Unknown provider 'xxx'`  
**Solution**: Use valid provider: `openai` or `mistral`

**Problem**: Slow responses  
**Solution**: Use `gpt-4o-mini` or enable streaming

**Problem**: High costs  
**Solution**: Monitor tokens with `count_tokens()`, use mini models

## Support

- GitHub Issues: [your-repo]
- Documentation: `ai_agents/components/llm_gateway/README.md`
- Examples: `examples/llm_gateway_example.py`


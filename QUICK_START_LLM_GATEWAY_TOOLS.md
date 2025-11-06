# Quick Start: LLM Gateway Tools

Get started with LLM Gateway Tools in 5 minutes! This guide shows you how to use LLM generation capabilities as tools for meta-reasoning, delegation, and specialized NLP tasks.

## Prerequisites

1. **Install Dependencies**
   ```bash
   pip install openai mistralai pydantic
   ```

2. **Set API Keys**
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export MISTRAL_API_KEY="your-mistral-key"  # Optional
   ```

## Quick Examples

### 1. Basic Text Generation (30 seconds)

```python
import asyncio
from ai_agents.components.tools.llm_gateway_tools import llm_generate

async def basic_example():
    result = await llm_generate(
        user_prompt="What are the three laws of robotics?",
        temperature=0.7
    )
    print(result['content'])

asyncio.run(basic_example())
```

**Output:**
```
The three laws of robotics, as formulated by Isaac Asimov, are:
1. A robot may not injure a human being...
2. A robot must obey orders given...
3. A robot must protect its own existence...
```

### 2. Structured Output (1 minute)

```python
from ai_agents.components.tools.llm_gateway_tools import llm_generate_structured

async def structured_example():
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "year": {"type": "integer"}
        }
    }
    
    result = await llm_generate_structured(
        user_prompt="Tell me about '1984' by George Orwell",
        schema=schema
    )
    print(result['structured_output'])

asyncio.run(structured_example())
```

**Output:**
```json
{
  "title": "1984",
  "author": "George Orwell",
  "year": 1949
}
```

### 3. Multi-Model Comparison (2 minutes)

```python
from ai_agents.components.tools.llm_gateway_tools import llm_multi_model_generate

async def multi_model_example():
    result = await llm_multi_model_generate(
        user_prompt="What is artificial intelligence in one sentence?",
        models=["openai:gpt-4o", "mistral:mistral-large-latest"]
    )
    
    for response in result['responses']:
        print(f"{response['model']}: {response['content']}\n")

asyncio.run(multi_model_example())
```

### 4. Token Counting (30 seconds)

```python
from ai_agents.components.tools.llm_gateway_tools import llm_count_tokens

async def token_example():
    text = "The quick brown fox jumps over the lazy dog"
    result = await llm_count_tokens(text=text)
    print(f"Tokens: {result['token_count']}")

asyncio.run(token_example())
```

### 5. Summarization (1 minute)

```python
from ai_agents.components.tools.llm_gateway_tools import llm_summarize

async def summarize_example():
    long_text = """
    Your long article or document here...
    Multiple paragraphs of content...
    """
    
    result = await llm_summarize(
        text=long_text,
        max_length=50  # Max 50 words
    )
    print(result['summary'])

asyncio.run(summarize_example())
```

### 6. Entity Extraction (1 minute)

```python
from ai_agents.components.tools.llm_gateway_tools import llm_extract_entities

async def entity_example():
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    result = await llm_extract_entities(
        text=text,
        entity_types=["person", "organization", "location"]
    )
    print(result['entities'])

asyncio.run(entity_example())
```

**Output:**
```json
{
  "person": ["Steve Jobs"],
  "organization": ["Apple Inc."],
  "location": ["Cupertino", "California"]
}
```

### 7. Translation (1 minute)

```python
from ai_agents.components.tools.llm_gateway_tools import llm_translate

async def translate_example():
    result = await llm_translate(
        text="Hello, how are you?",
        target_language="Spanish"
    )
    print(result['translation'])

asyncio.run(translate_example())
```

**Output:**
```
Hola, ¿cómo estás?
```

## Using with Agents

### Step 1: Create Tool Registry

```python
from ai_core.interfaces.tool_registry import ToolRegistry
from ai_agents.components.tools.llm_gateway_tools import get_all_llm_gateway_tools

# Create registry
registry = ToolRegistry()

# Register all LLM gateway tools
tool_configs = get_all_llm_gateway_tools()
for config in tool_configs:
    registry.register_direct_tool(
        name=config["name"],
        function=config["function"],
        description=config["description"],
        json_schema=config["json_schema"],
        category=config.get("category"),
        tags=config.get("tags", []),
    )

print(f"Registered {len(tool_configs)} tools")
```

### Step 2: Create Agent with Tools

```python
from ai_agents.components.base_agent import BaseAgent

agent = BaseAgent(
    name="meta_agent",
    model="openai:gpt-4o",
    system_prompt="""
    You are an agent with access to LLM tools.
    You can delegate tasks, compare models, and use specialized LLM functions.
    """,
    tool_registry=registry
)
```

### Step 3: Use Agent

```python
result = await agent.arun(
    """
    Analyze this text: 'Apple Inc. was founded by Steve Jobs.'
    1. Extract entities
    2. Summarize it
    3. Translate the summary to French
    """
)

print(result.output_text)
```

## Load from Configuration

### Create YAML Config

```yaml
# my_tools.yaml
tools:
  - name: llm_generate
    type: direct
    module: ai_agents.components.tools.llm_gateway_tools
    function: llm_generate
    description: Generate text using an LLM
    category: llm
    tags: [generation]
    schema:
      type: object
      properties:
        user_prompt:
          type: string
      required: [user_prompt]
```

### Load Config

```python
from ai_core.interfaces.tool_loader import ToolLoader

loader = ToolLoader(registry)
loader.load_from_yaml("my_tools.yaml")
```

## Meta-Reasoning Example

Use one LLM to decide which LLM to use for a task:

```python
async def meta_reasoning():
    # Step 1: Planner decides which model to use
    decision = await llm_generate(
        user_prompt="""
        I need to write a creative poem. 
        Which model should I use: gpt-4o or mistral-large-latest?
        Why? Be brief.
        """,
        temperature=0.3  # Low temp for consistent decisions
    )
    
    print(f"Decision: {decision['content']}\n")
    
    # Step 2: Execute with chosen model
    result = await llm_generate(
        user_prompt="Write a haiku about coding",
        model="mistral:mistral-large-latest",  # Based on decision
        temperature=0.9  # High temp for creativity
    )
    
    print(f"Result: {result['content']}")

asyncio.run(meta_reasoning())
```

## Available Tools Summary

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `llm_generate` | General text generation | `user_prompt`, `model`, `temperature` |
| `llm_generate_structured` | Structured output | `user_prompt`, `schema` |
| `llm_count_tokens` | Count tokens | `text`, `model` |
| `llm_multi_model_generate` | Compare models | `user_prompt`, `models` |
| `llm_summarize` | Summarize text | `text`, `max_length` |
| `llm_extract_entities` | Extract entities | `text`, `entity_types` |
| `llm_translate` | Translate text | `text`, `target_language` |

## Common Use Cases

### 1. Document Processing Pipeline

```python
async def process_document(text):
    # Step 1: Summarize
    summary = await llm_summarize(text=text, max_length=100)
    
    # Step 2: Extract entities
    entities = await llm_extract_entities(
        text=summary['summary'],
        entity_types=["person", "organization", "location"]
    )
    
    # Step 3: Translate
    translation = await llm_translate(
        text=summary['summary'],
        target_language="Spanish"
    )
    
    return {
        "summary": summary['summary'],
        "entities": entities['entities'],
        "spanish_summary": translation['translation']
    }
```

### 2. Multi-Model Consensus

```python
async def get_consensus(question):
    result = await llm_multi_model_generate(
        user_prompt=question,
        models=["openai:gpt-4o", "mistral:mistral-large-latest"]
    )
    
    # Compare responses
    for i, resp in enumerate(result['responses'], 1):
        print(f"Model {i}: {resp['content']}\n")
```

### 3. Cost-Aware Generation

```python
async def generate_with_budget(prompt, max_tokens_budget=1000):
    # Check cost first
    token_check = await llm_count_tokens(text=prompt)
    
    if token_check['token_count'] > max_tokens_budget:
        print("Prompt too long, summarizing first...")
        summary = await llm_summarize(text=prompt, max_length=100)
        prompt = summary['summary']
    
    # Generate
    result = await llm_generate(user_prompt=prompt)
    return result['content']
```

## Model Selection Guide

| Model | Best For | Cost | Speed |
|-------|----------|------|-------|
| `openai:gpt-4o` | Complex reasoning, analysis | $$$ | Medium |
| `openai:gpt-4o-mini` | Simple tasks, quick responses | $ | Fast |
| `mistral:mistral-large-latest` | Balanced tasks, multilingual | $$ | Fast |
| `mistral:mistral-small-latest` | Lightweight tasks | $ | Very Fast |

## Temperature Guide

| Temperature | Use For | Examples |
|-------------|---------|----------|
| 0.0 - 0.3 | Factual, deterministic | Classification, extraction, factual Q&A |
| 0.4 - 0.7 | Balanced | General Q&A, summarization, analysis |
| 0.8 - 1.0 | Creative | Story writing, brainstorming, poetry |

## Environment Setup

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional (for multi-model features)
export MISTRAL_API_KEY="..."

# Optional (defaults)
export DEFAULT_LLM_MODEL="openai:gpt-4o"
export DEFAULT_LLM_TEMPERATURE="0.7"
```

## Troubleshooting

### "No API key found"
**Solution**: Set `OPENAI_API_KEY` or `MISTRAL_API_KEY`

### "Unknown provider"
**Solution**: Use format `provider:model` (e.g., `openai:gpt-4o`)

### "Token limit exceeded"
**Solution**: Use `llm_count_tokens` before generation

### Import errors
**Solution**: Check Python path and install dependencies

## Next Steps

1. **Run Examples**: `python examples/llm_gateway_tools_example.py`
2. **Read Full Docs**: See `LLM_GATEWAY_TOOLS_README.md`
3. **Explore LLM Gateway**: See `QUICK_START_LLM_GATEWAY.md`
4. **Create Custom Workflows**: Combine tools for your use case

## Complete Example Script

Save as `test_llm_tools.py`:

```python
import asyncio
from ai_agents.components.tools.llm_gateway_tools import (
    llm_generate,
    llm_summarize,
    llm_extract_entities,
)

async def main():
    # Generate
    result = await llm_generate(
        user_prompt="Explain AI in one paragraph"
    )
    print(f"Generation:\n{result['content']}\n")
    
    # Summarize
    summary = await llm_summarize(
        text=result['content'],
        max_length=20
    )
    print(f"Summary:\n{summary['summary']}\n")
    
    # Extract entities
    entities = await llm_extract_entities(
        text=result['content']
    )
    print(f"Entities:\n{entities['entities']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run: `python test_llm_tools.py`

## Resources

- **Full Documentation**: `ai_agents/components/tools/LLM_GATEWAY_TOOLS_README.md`
- **Examples**: `examples/llm_gateway_tools_example.py`
- **Configuration**: `examples/configs/llm_gateway_tools_config.yaml`
- **LLM Gateway Docs**: `QUICK_START_LLM_GATEWAY.md`

---

**Ready to build meta-reasoning agents!** 🚀

For questions or issues, refer to the full documentation or check the examples directory.


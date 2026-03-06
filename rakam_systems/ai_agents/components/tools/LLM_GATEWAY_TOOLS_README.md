# LLM Gateway Tools

A comprehensive set of tools that expose LLM Gateway generation functions, enabling agents to use LLM capabilities as tools for meta-reasoning, delegation, and specialized NLP tasks.

## Overview

The LLM Gateway Tools bridge the gap between agents and LLM generation capabilities, allowing agents to:

- **Meta-Reasoning**: Agents can use LLMs to reason about complex problems
- **Task Delegation**: Delegate subtasks to specialized models
- **Multi-Model Workflows**: Compare outputs or build consensus across models
- **Specialized Operations**: Use LLMs for summarization, entity extraction, translation, etc.

## Available Tools

### Core Generation Tools

#### 1. `llm_generate`
Generate text using an LLM through the gateway.

**Parameters:**
- `user_prompt` (required): The main prompt/question for the LLM
- `system_prompt` (optional): System prompt to set context/behavior
- `model` (optional): Model string (e.g., "openai:gpt-4o", "mistral:mistral-large-latest")
- `temperature` (optional): Temperature for generation (0.0-1.0)
- `max_tokens` (optional): Maximum tokens to generate

**Returns:**
- `content`: The generated text
- `model`: Model used for generation
- `usage`: Token usage information
- `finish_reason`: Why generation stopped
- `metadata`: Additional metadata

**Use Cases:**
- Multi-step reasoning
- Delegation to specialized models
- Meta-reasoning workflows

**Example:**
```python
from ai_agents.components.tools.llm_gateway_tools import llm_generate

result = await llm_generate(
    user_prompt="Explain quantum entanglement",
    system_prompt="You are a physics expert",
    model="openai:gpt-4o",
    temperature=0.7
)
print(result['content'])
```

#### 2. `llm_generate_structured`
Generate structured output conforming to a JSON schema.

**Parameters:**
- `user_prompt` (required): The main prompt/question
- `schema` (required): JSON schema defining expected output structure
- `system_prompt` (optional): System prompt
- `model` (optional): Model string
- `temperature` (optional): Temperature (0.0-1.0)
- `max_tokens` (optional): Maximum tokens

**Returns:**
- `structured_output`: The parsed structured output
- `raw_content`: The raw text response
- `model`: Model used
- `usage`: Token usage information

**Use Cases:**
- Extracting structured data
- Ensuring consistent output format
- Type-safe responses

**Example:**
```python
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
```

#### 3. `llm_count_tokens`
Count tokens in text using the LLM gateway's tokenizer.

**Parameters:**
- `text` (required): Text to count tokens for
- `model` (optional): Model string for tokenization

**Returns:**
- `token_count`: Number of tokens
- `model`: Model used
- `text_length`: Character length

**Use Cases:**
- Checking prompt lengths before generation
- Estimating API costs
- Managing context windows

**Example:**
```python
result = await llm_count_tokens(
    text="The quick brown fox jumps over the lazy dog",
    model="openai:gpt-4o"
)
print(f"Tokens: {result['token_count']}")
```

#### 4. `llm_multi_model_generate`
Generate responses from multiple models in parallel.

**Parameters:**
- `user_prompt` (required): The main prompt
- `models` (required): List of model strings
- `system_prompt` (optional): System prompt
- `temperature` (optional): Temperature (0.0-1.0)
- `max_tokens` (optional): Maximum tokens

**Returns:**
- `responses`: List of responses from each model
- `model_count`: Number of models queried

**Use Cases:**
- Comparing outputs across models
- Building consensus
- Model ensemble approaches
- A/B testing prompts

**Example:**
```python
result = await llm_multi_model_generate(
    user_prompt="What is the meaning of life?",
    models=["openai:gpt-4o", "mistral:mistral-large-latest"]
)

for response in result['responses']:
    print(f"{response['model']}: {response['content']}")
```

### Specialized NLP Tools

#### 5. `llm_summarize`
Summarize text using an LLM.

**Parameters:**
- `text` (required): Text to summarize
- `model` (optional): Model string
- `max_length` (optional): Maximum length for summary in words

**Returns:**
- `summary`: The generated summary
- `original_length`: Length of original text (words)
- `summary_length`: Length of summary (words)
- `model`: Model used
- `usage`: Token usage

**Example:**
```python
result = await llm_summarize(
    text="Long article text...",
    max_length=100
)
print(result['summary'])
```

#### 6. `llm_extract_entities`
Extract named entities from text.

**Parameters:**
- `text` (required): Text to extract entities from
- `entity_types` (optional): List of entity types to extract (e.g., ["person", "organization", "location"])
- `model` (optional): Model string

**Returns:**
- `entities`: Extracted entities (as JSON)
- `model`: Model used
- `usage`: Token usage

**Example:**
```python
result = await llm_extract_entities(
    text="Apple Inc. was founded by Steve Jobs in Cupertino.",
    entity_types=["person", "organization", "location"]
)
print(result['entities'])
```

#### 7. `llm_translate`
Translate text using an LLM.

**Parameters:**
- `text` (required): Text to translate
- `target_language` (required): Target language (e.g., "Spanish", "French")
- `source_language` (optional): Source language (auto-detected if not specified)
- `model` (optional): Model string

**Returns:**
- `translation`: The translated text
- `source_language`: Source language used
- `target_language`: Target language
- `model`: Model used
- `usage`: Token usage

**Example:**
```python
result = await llm_translate(
    text="Hello, how are you?",
    target_language="Spanish"
)
print(result['translation'])
```

## Usage Patterns

### 1. Direct Usage

Use the tools directly in your code:

```python
from ai_agents.components.tools.llm_gateway_tools import llm_generate

async def my_function():
    result = await llm_generate(
        user_prompt="What is AI?",
        temperature=0.7
    )
    return result['content']
```

### 2. Registration with Tool Registry

Register tools for use with agents:

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
```

### 3. Configuration-Based Loading

Load tools from YAML configuration:

```yaml
# llm_gateway_tools_config.yaml
tools:
  - name: llm_generate
    type: direct
    module: ai_agents.components.tools.llm_gateway_tools
    function: llm_generate
    description: Generate text using an LLM
    category: llm
    tags: [generation, llm]
    schema:
      type: object
      properties:
        user_prompt:
          type: string
          description: The main prompt
      required: [user_prompt]
```

```python
from ai_core.interfaces.tool_registry import ToolRegistry
from ai_core.interfaces.tool_loader import ToolLoader

registry = ToolRegistry()
loader = ToolLoader(registry)
loader.load_from_yaml("llm_gateway_tools_config.yaml")
```

### 4. Use with Agents

Create agents with LLM gateway tools:

```python
from ai_agents.components.base_agent import BaseAgent
from ai_core.interfaces.tool_registry import ToolRegistry

# Create registry and register tools
registry = ToolRegistry()
# ... register tools ...

# Create agent with tools
agent = BaseAgent(
    name="meta_reasoning_agent",
    model="openai:gpt-4o",
    system_prompt="You are an agent with LLM tool access",
    tool_registry=registry
)

# Agent can now use LLM tools
result = await agent.arun(
    "Summarize this article and extract key entities: ..."
)
```

## Advanced Patterns

### Meta-Reasoning

Use one LLM to reason about which LLM to use for a task:

```python
# Step 1: Ask planner LLM which model to use
decision = await llm_generate(
    user_prompt="Which model is best for creative writing?",
    system_prompt="You are an AI model selection expert",
    temperature=0.3
)

# Step 2: Use recommended model for the actual task
result = await llm_generate(
    user_prompt="Write a creative story about...",
    model="mistral:mistral-large-latest",  # Based on recommendation
    temperature=0.9
)
```

### Multi-Model Consensus

Get consensus from multiple models:

```python
result = await llm_multi_model_generate(
    user_prompt="Is this statement true: ...",
    models=["openai:gpt-4o", "mistral:mistral-large-latest"]
)

# Analyze responses for consensus
responses = [r['content'] for r in result['responses']]
```

### Hierarchical Task Decomposition

Agent breaks down complex tasks:

```python
# Agent with LLM tools can:
# 1. Use llm_generate to break down a complex task
# 2. Use llm_summarize on long inputs
# 3. Use llm_extract_entities to find key information
# 4. Use llm_translate to handle multilingual content
# 5. Use llm_multi_model_generate to verify results
```

### Cost Optimization

Check token counts before generation:

```python
# Check token count
token_result = await llm_count_tokens(
    text=my_long_prompt,
    model="openai:gpt-4o"
)

# Only proceed if within budget
if token_result['token_count'] < 1000:
    result = await llm_generate(
        user_prompt=my_long_prompt,
        model="openai:gpt-4o"
    )
```

## Configuration

### Model Selection

Tools respect the model string format:
- `"openai:gpt-4o"` - OpenAI GPT-4o
- `"openai:gpt-4o-mini"` - OpenAI GPT-4o-mini
- `"mistral:mistral-large-latest"` - Mistral Large
- `"mistral:mistral-small-latest"` - Mistral Small

### Environment Variables

Tools use the LLM Gateway Factory which respects:
- `DEFAULT_LLM_MODEL` - Default model if not specified
- `DEFAULT_LLM_TEMPERATURE` - Default temperature
- `DEFAULT_LLM_PROVIDER` - Default provider
- `OPENAI_API_KEY` - OpenAI API key
- `MISTRAL_API_KEY` - Mistral API key

### Temperature Guidelines

- `0.0-0.3`: Factual, deterministic tasks (classification, extraction)
- `0.4-0.7`: Balanced tasks (general Q&A, summarization)
- `0.8-1.0`: Creative tasks (story writing, brainstorming)

## Examples

See `examples/llm_gateway_tools_example.py` for comprehensive examples including:
- Basic generation
- Structured output
- Token counting
- Multi-model comparison
- Summarization
- Entity extraction
- Translation
- Meta-reasoning workflows
- Agent integration

Run examples:
```bash
cd examples
python llm_gateway_tools_example.py
```

## Integration with Other Systems

### With Base Agent

```python
agent = BaseAgent(
    model="openai:gpt-4o",
    tool_registry=registry  # Contains LLM gateway tools
)
```

### With Tool Invoker

```python
from ai_core.interfaces.tool_invoker import ToolInvoker

invoker = ToolInvoker(registry)
result = await invoker.invoke_tool(
    "llm_generate",
    {"user_prompt": "Hello"}
)
```

### With MCP Servers

LLM gateway tools can be exposed via MCP servers for remote access.

## Best Practices

1. **Model Selection**: Choose the right model for the task
   - GPT-4o for complex reasoning
   - GPT-4o-mini for simple tasks (cost-effective)
   - Mistral Large for balanced performance
   - Mistral Small for lightweight tasks

2. **Temperature Control**: Adjust based on task type
   - Low for factual tasks
   - High for creative tasks

3. **Token Management**: Always check token counts for large inputs

4. **Error Handling**: Handle API errors and rate limits gracefully

5. **Caching**: Consider caching results for repeated queries

6. **Cost Optimization**: 
   - Use token counting before generation
   - Choose appropriate models for task complexity
   - Set reasonable max_tokens limits

7. **Security**: 
   - Don't expose sensitive information in prompts
   - Validate and sanitize user inputs
   - Use appropriate system prompts to constrain behavior

## Troubleshooting

### API Key Issues
```
Error: No API key found
```
**Solution**: Set `OPENAI_API_KEY` or `MISTRAL_API_KEY` environment variable

### Model Not Found
```
Error: Unknown provider 'xyz'
```
**Solution**: Use supported model strings (openai:*, mistral:*)

### Token Limits
```
Error: Token limit exceeded
```
**Solution**: Use `llm_count_tokens` to check before generation

### Import Errors
```
Error: Module not found
```
**Solution**: Ensure all dependencies are installed and paths are correct

## Contributing

To add new LLM gateway tools:

1. Add function to `llm_gateway_tools.py`
2. Follow the async function signature pattern
3. Return structured dictionaries
4. Add to `get_all_llm_gateway_tools()` configuration list
5. Update this README with documentation
6. Add example usage to `llm_gateway_tools_example.py`
7. Add to YAML configuration in `examples/configs/`

## Related Documentation

- [LLM Gateway System](../llm_gateway/README.md)
- [Tool System Documentation](../../../../ai_core/interfaces/tool_registry.py)
- [Base Agent Documentation](../base_agent.py)
- [Configuration System](../../../../ai_core/CONFIG_SYSTEM_README.md)


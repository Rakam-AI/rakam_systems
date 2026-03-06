# Configuration System - Quick Start Guide

## 5-Minute Setup

### 1. Create a Configuration File

Save as `my_agent_config.yaml`:

```yaml
version: "1.0"

prompts:
  helpful_assistant:
    name: "helpful_assistant"
    system_prompt: "You are a helpful AI assistant."
    skills:
      - "Information retrieval"
      - "Task automation"

tools: {}  # Add tools here if needed

agents:
  my_agent:
    name: "my_agent"
    model_config:
      model: "openai:gpt-4o"
      temperature: 0.7
      max_tokens: 2000
      parallel_tool_calls: true
    prompt_config: "helpful_assistant"
    tools: []
    enable_tracking: true
    tracking_output_dir: "./agent_tracking"
```

### 2. Use the Agent

```python
import asyncio
from ai_core import ConfigurationLoader

async def main():
    # Load configuration
    loader = ConfigurationLoader()
    loader.load_from_yaml("my_agent_config.yaml")
    
    # Create agent
    agent = loader.create_agent("my_agent")
    
    # Use agent (tracking happens automatically)
    result = await agent.arun("What is 2 + 2?")
    print(result.output_text)
    
    # Export tracking data to CSV
    csv_path = agent.export_tracking_data(format='csv')
    print(f"Tracking data saved to: {csv_path}")
    
    # Get statistics
    stats = agent.get_tracking_statistics()
    print(f"Total calls: {stats['total_calls']}")
    print(f"Success rate: {stats['success_rate']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Check the Results

Your tracking data is now in a CSV file with columns:
- `input_text`, `output_text`
- `duration_seconds`
- `success`
- `evaluation_score` (empty, for your evaluation)
- `evaluation_notes` (empty, for your notes)
- Model settings, token usage, etc.

## Adding Tools

### Define a Tool in Config

```yaml
tools:
  get_weather:
    name: "get_weather"
    type: "direct"
    module: "my_tools"
    function: "get_weather"
    description: "Get weather for a location"
    schema:
      type: "object"
      properties:
        location:
          type: "string"
          description: "City name"
      required: ["location"]
```

### Create the Tool Function

In `my_tools.py`:

```python
async def get_weather(location: str) -> dict:
    """Get weather for a location."""
    # Your implementation here
    return {
        "location": location,
        "temperature": 72,
        "condition": "sunny"
    }
```

### Use the Tool

```yaml
agents:
  my_agent:
    # ... other config
    tools:
      - "get_weather"  # Reference the tool
```

## Common Patterns

### Multiple Agents with Shared Prompts

```yaml
prompts:
  base_prompt:
    name: "base_prompt"
    system_prompt: "You are helpful."

agents:
  agent1:
    # ...
    prompt_config: "base_prompt"
  
  agent2:
    # ...
    prompt_config: "base_prompt"
```

### Different Models for Different Tasks

```yaml
agents:
  fast_agent:
    model_config:
      model: "openai:gpt-4o-mini"  # Faster, cheaper
      temperature: 0.5
  
  powerful_agent:
    model_config:
      model: "openai:gpt-4o"  # More capable
      temperature: 0.7
```

### Sequential vs Parallel Tool Calls

```yaml
# Parallel (faster, for independent operations)
model_config:
  parallel_tool_calls: true

# Sequential (for dependent operations)
model_config:
  parallel_tool_calls: false
```

## Programmatic Configuration

Don't want YAML? Create config in code:

```python
from ai_core import ConfigurationLoader
from ai_core.config_schema import (
    AgentConfigSchema,
    ModelConfigSchema,
    PromptConfigSchema,
)

config_dict = {
    "version": "1.0",
    "prompts": {
        "simple": {
            "name": "simple",
            "system_prompt": "You are helpful.",
        }
    },
    "tools": {},
    "agents": {
        "my_agent": {
            "name": "my_agent",
            "model_config": {
                "model": "openai:gpt-4o",
                "temperature": 0.7,
            },
            "prompt_config": "simple",
            "tools": [],
            "enable_tracking": True,
        }
    }
}

loader = ConfigurationLoader()
loader.load_from_dict(config_dict)
agent = loader.create_agent("my_agent")
```

## Evaluation Workflow

1. **Configure**: Set up agent with tracking enabled
2. **Run**: Execute agent on test cases
3. **Export**: Get CSV with all data
4. **Evaluate**: Fill in `evaluation_score` and `evaluation_notes` columns
5. **Analyze**: Use Excel/Python to analyze results
6. **Iterate**: Update configuration based on findings

## Validation

Before deploying, validate your config:

```python
loader = ConfigurationLoader()
loader.load_from_yaml("config.yaml")

is_valid, errors = loader.validate_config()

if not is_valid:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid!")
```

## Session Management

For organized tracking:

```python
# Start a session
manager = agent.get_tracking_manager()
session_id = manager.start_session(agent.name)

# Make calls
for query in queries:
    await agent.arun(query)

# End session
manager.end_session(session_id)

# Export just this session
csv_path = manager.export_to_csv(
    filename="session.csv",
    session_id=session_id
)
```

## Tips

### Temperature Settings
- **0.1-0.3**: Factual, deterministic (math, facts)
- **0.7-0.8**: Balanced (general purpose)
- **0.9-1.0**: Creative (writing, brainstorming)

### Tracking Best Practices
- Always enable in production
- Use separate directories per agent
- Export regularly (daily/weekly)
- Archive old data

### Performance
- Tracking overhead: ~1-5ms (negligible)
- No impact when disabled
- Export happens in background

## Troubleshooting

### "Module not found"
```python
# Make sure module is importable
import my_module  # Should work before using in config
```

### "Tracking not working"
```yaml
# Check these settings
enable_tracking: true  # Must be true
tracking_output_dir: "./tracking"  # Must be writable
```

### "Configuration invalid"
```python
# Use validation to find issues
is_valid, errors = loader.validate_config()
print(errors)
```

## Examples

Complete working examples:
- `examples/config_system_example.py`: Full walkthrough
- `examples/configs/complete_agent_config.yaml`: Comprehensive config

## Documentation

- **Quick Start**: This file
- **Complete Guide**: `ai_core/CONFIG_SYSTEM_README.md`
- **Implementation**: `CONFIGURATION_SYSTEM_IMPLEMENTATION.md`

## Next Steps

1. ✅ Create your config file
2. ✅ Load and validate it
3. ✅ Create your agent
4. ✅ Run some queries
5. ✅ Export tracking data
6. ✅ Evaluate and iterate!

That's it! You're ready to build configurable, trackable AI agents. 🚀


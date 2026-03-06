# Agent Configuration and Tracking System

A comprehensive configuration and tracking system for AI agents, enabling declarative agent setup, automatic input/output tracking, and evaluation data generation.

## Overview

This system provides three main components:

1. **Configuration System**: Define agents, tools, prompts, and models in YAML
2. **Tracking System**: Automatically capture inputs, outputs, and performance metrics
3. **Export System**: Generate CSV/JSON files for evaluation and analysis

## Features

### Configuration System

- **Declarative Setup**: Define agents in YAML configuration files
- **Reusable Components**: Share prompts and tools across multiple agents
- **Dynamic Loading**: Load Python modules and classes at runtime
- **Validation**: Catch configuration errors before runtime
- **Pydantic Schemas**: Type-safe configuration with validation

### Tracking System

- **Automatic Tracking**: Decorator-based method tracking
- **Input/Output Capture**: Records all method calls with full context
- **Performance Metrics**: Duration, success rate, token usage
- **Session Management**: Group related calls into sessions
- **No Performance Impact**: Minimal overhead when disabled

### Export System

- **CSV Export**: Ready for spreadsheet analysis
- **JSON Export**: Structured data for programmatic access
- **Evaluation Fields**: Pre-formatted for evaluation workflows
- **Statistics**: Aggregate metrics and analysis

## Quick Start

### 1. Create a Configuration File

```yaml
# agent_config.yaml
version: "1.0"

prompts:
  helpful_assistant:
    name: "helpful_assistant"
    system_prompt: "You are a helpful AI assistant."
    skills:
      - "Information retrieval"
      - "Task automation"

tools:
  get_weather:
    name: "get_weather"
    type: "direct"
    module: "my_tools.weather"
    function: "get_weather"
    description: "Get current weather"
    schema:
      type: "object"
      properties:
        location:
          type: "string"
      required: ["location"]

agents:
  my_agent:
    name: "my_agent"
    model_config:
      model: "openai:gpt-4o"
      temperature: 0.7
      parallel_tool_calls: true
    prompt_config: "helpful_assistant"
    tools:
      - "get_weather"
    enable_tracking: true
    tracking_output_dir: "./tracking"
```

### 2. Load and Use the Agent

```python
from ai_core import ConfigurationLoader

# Load configuration
loader = ConfigurationLoader()
loader.load_from_yaml("agent_config.yaml")

# Create agent
agent = loader.create_agent("my_agent")

# Use agent (tracking is automatic)
result = await agent.arun("What's the weather in Paris?")

# Export tracking data
csv_path = agent.export_tracking_data(format='csv')
```

### 3. Analyze Tracking Data

The exported CSV includes:

- Call ID, timestamps, duration
- Input text, output text
- Success/failure status
- Model settings (temperature, max_tokens, etc.)
- Token usage (prompt, completion, total)
- Tool call counts
- Evaluation fields (score, notes)

## Configuration Schema

### Root Structure

```yaml
version: "1.0"
global_settings: {}
prompts: {}
tools: {}
agents: {}
```

### Prompt Configuration

```yaml
prompts:
  prompt_name:
    name: "prompt_name"
    description: "Optional description"
    system_prompt: "Your system prompt here"
    skills:
      - "Skill 1"
      - "Skill 2"
    tags:
      - "tag1"
    examples:
      - user: "Example input"
        assistant: "Example output"
```

### Tool Configuration

**Direct Tools** (run locally):

```yaml
tools:
  tool_name:
    name: "tool_name"
    type: "direct"
    module: "python.module.path"
    function: "function_name"
    description: "What the tool does"
    category: "utility"
    tags: ["tag1", "tag2"]
    schema:
      type: "object"
      properties:
        param1:
          type: "string"
          description: "Parameter description"
      required: ["param1"]
```

**MCP Tools** (remote via MCP protocol):

```yaml
tools:
  tool_name:
    name: "tool_name"
    type: "mcp"
    mcp_server: "server_name"
    mcp_tool_name: "remote_tool_name"
    description: "What the tool does"
    category: "external"
```

### Agent Configuration

```yaml
agents:
  agent_name:
    name: "agent_name"
    description: "Agent description"
    
    # Model settings
    model_config:
      model: "openai:gpt-4o"
      temperature: 0.7
      max_tokens: 2000
      parallel_tool_calls: true
      extra_settings:
        top_p: 0.9
    
    # Prompt (reference or inline)
    prompt_config: "prompt_name"
    
    # Tools (references)
    tools:
      - "tool_name"
    
    # Tracking
    enable_tracking: true
    tracking_output_dir: "./tracking"
    
    # Other settings
    stateful: false
    deps_type: "my.module.DepsClass"
    metadata:
      custom_field: "value"
```

## API Reference

### ConfigurationLoader

```python
from ai_core import ConfigurationLoader

loader = ConfigurationLoader()

# Load from file
config = loader.load_from_yaml("config.yaml")

# Load from dict
config = loader.load_from_dict(config_dict)

# Validate configuration
is_valid, errors = loader.validate_config()

# Create single agent
agent = loader.create_agent("agent_name")

# Create all agents
agents = loader.create_all_agents()
```

### TrackingMixin

Agents inherit tracking capabilities:

```python
# Enable/disable tracking
agent.enable_tracking("./custom_dir")
agent.disable_tracking()

# Get tracking manager
manager = agent.get_tracking_manager()

# Export data
csv_path = agent.export_tracking_data(format='csv')
json_path = agent.export_tracking_data(format='json')

# Get statistics
stats = agent.get_tracking_statistics()
```

### TrackingManager

Direct access to tracking:

```python
from ai_core import get_tracking_manager

manager = get_tracking_manager("./tracking")

# Session management
session_id = manager.start_session("agent_name")
manager.end_session(session_id)

# Export
csv_path = manager.export_to_csv()
json_path = manager.export_to_json()

# Statistics
stats = manager.get_statistics()
```

### track_method Decorator

For custom agent classes:

```python
from ai_core import track_method

class MyAgent:
    @track_method()
    async def my_method(self, input_data):
        # Method implementation
        return result
```

## Pydantic Schemas

All configuration uses Pydantic for validation:

```python
from ai_core import (
    AgentConfigSchema,
    ToolConfigSchema,
    ModelConfigSchema,
    PromptConfigSchema,
    ConfigFileSchema,
    MethodInputSchema,
    MethodOutputSchema,
    MethodCallRecordSchema,
    TrackingSessionSchema,
)

# Use for type hints, validation, or programmatic creation
config = AgentConfigSchema(
    name="my_agent",
    model_config=ModelConfigSchema(
        model="openai:gpt-4o",
        temperature=0.7,
    ),
    prompt_config="my_prompt",
    tools=[],
)
```

## CSV Export Format

The exported CSV includes these columns:

| Column | Description |
|--------|-------------|
| `call_id` | Unique identifier for the call |
| `agent_name` | Name of the agent |
| `method_name` | Method that was called |
| `started_at` | Start timestamp (ISO format) |
| `completed_at` | Completion timestamp |
| `duration_seconds` | Execution time |
| `success` | True if successful |
| `input_text` | Input text |
| `output_text` | Output text |
| `error` | Error message if failed |
| `evaluation_score` | (Empty, for manual evaluation) |
| `evaluation_notes` | (Empty, for manual notes) |
| `model` | Model used |
| `temperature` | Temperature setting |
| `max_tokens` | Max tokens setting |
| `parallel_tool_calls` | Parallel setting |
| `tool_calls_count` | Number of tool calls |
| `usage_prompt_tokens` | Prompt tokens used |
| `usage_completion_tokens` | Completion tokens used |
| `usage_total_tokens` | Total tokens used |

## Best Practices

### Configuration

1. **Organize by Environment**: Use different config files for dev/staging/production
2. **Reuse Components**: Define prompts and tools once, reference many times
3. **Version Control**: Keep configs in git, use semantic versioning
4. **Document Prompts**: Include descriptions and examples for each prompt
5. **Validate Early**: Run validation before deploying

### Tracking

1. **Always Enable in Production**: Track all production agent calls
2. **Separate Directories**: Use different tracking dirs per agent
3. **Regular Exports**: Schedule periodic CSV exports for analysis
4. **Monitor Statistics**: Check success rates and durations regularly
5. **Clean Up Old Data**: Archive or delete old tracking data

### Tools

1. **Clear Schemas**: Provide detailed JSON schemas for all tools
2. **Error Handling**: Tools should return meaningful error messages
3. **Categories and Tags**: Organize tools for easy filtering
4. **Document Side Effects**: Note if tools modify state or external systems
5. **Test Independently**: Test tools outside of agents first

### Model Settings

1. **Temperature by Task**:
   - 0.1-0.3: Factual, deterministic tasks
   - 0.7-0.8: Balanced, general purpose
   - 0.9-1.0: Creative tasks

2. **Parallel Tool Calls**:
   - `true`: Independent operations (faster)
   - `false`: Dependent operations (sequential)

3. **Token Limits**: Set appropriate limits based on task complexity

## Examples

See the `examples/` directory for comprehensive examples:

- `config_system_example.py`: Complete walkthrough of all features
- `configs/complete_agent_config.yaml`: Full configuration example

## Evaluation Workflow

1. **Configure Agent**: Set up agent with tracking enabled
2. **Run Tests**: Execute agent on test cases
3. **Export Data**: Generate CSV with tracking data
4. **Evaluate**: Fill in `evaluation_score` and `evaluation_notes` columns
5. **Analyze**: Use spreadsheet or Python to analyze results
6. **Iterate**: Update configuration based on findings

## Troubleshooting

### Configuration Errors

```python
# Use validation to find errors
is_valid, errors = loader.validate_config()
for error in errors:
    print(error)
```

### Tracking Not Working

Check:
1. `enable_tracking: true` in config
2. Agent has `_tracking_enabled` attribute
3. Methods are decorated with `@track_method()`
4. Directory permissions are correct

### Import Errors

Ensure modules are importable:
```python
# Test manually
import my_module
function = my_module.my_function
```

### Performance Issues

- Tracking overhead is minimal (~1-5ms per call)
- Export only when needed (not on every call)
- Use batch operations for analysis
- Archive old tracking data regularly

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Configuration Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ YAML Config  │→ │    Pydantic  │→ │  Agent       │      │
│  │ Files        │  │    Schemas   │  │  Instances   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Agent Layer                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  BaseAgent (with TrackingMixin)                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │ @track   │→ │  Method  │→ │  TrackingManager │  │   │
│  │  │ decorator│  │  Call    │  │                  │  │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   In-Memory  │→ │     CSV      │  │     JSON     │      │
│  │   Records    │  │    Export    │  │    Export    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Contributing

When extending the configuration system:

1. Update Pydantic schemas in `config_schema.py`
2. Update loader logic in `config_loader.py`
3. Add tests for new functionality
4. Update this documentation
5. Add examples demonstrating new features

## License

See main project LICENSE file.


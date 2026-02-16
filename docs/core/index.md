# Rakam System Core

The core package of Rakam Systems providing foundational interfaces, base components, and utilities.

## Overview

`rakam-systems-core` is the foundation of the Rakam Systems framework. It provides:

- **Base Component**: Abstract base class with lifecycle management
- **Interfaces**: Standard interfaces for agents, tools, vector stores, embeddings, and loaders
- **Configuration System**: YAML/JSON configuration loading and validation
- **Tracking System**: Input/output tracking for debugging and evaluation
- **Logging Utilities**: Structured logging with color support

This package is required by both `rakam-systems-agent` and `rakam-systems-vectorstore`.

## Installation

```bash
pip install -e ./rakam-systems-core
```

## Key Components

### BaseComponent

All components extend `BaseComponent` which provides:

- Lifecycle management with `setup()` and `shutdown()` methods
- Auto-initialization via `__call__`
- Context manager support
- Built-in evaluation harness

```python
from rakam_systems_core.ai_core.base import BaseComponent

class MyComponent(BaseComponent):
    def setup(self):
        super().setup()
        # Initialize resources

    def shutdown(self):
        # Clean up resources
        super().shutdown()

    def run(self, *args, **kwargs):
        # Main logic
        pass
```

### Interfaces

Standard interfaces for building AI systems:

- **AgentComponent**: AI agents with sync/async support
- **ToolComponent**: Callable tools for agents
- **LLMGateway**: LLM provider abstraction
- **VectorStore**: Vector storage interface
- **EmbeddingModel**: Text embedding interface
- **Loader**: Document loading interface
- **Chunker**: Text chunking interface

```python
from rakam_systems_core.ai_core.interfaces.agent import AgentComponent
from rakam_systems_core.ai_core.interfaces.tool import ToolComponent
from rakam_systems_core.ai_core.interfaces.vectorstore import VectorStore
```

### Configuration System

Load and validate configurations from YAML files:

```python
from rakam_systems_core.ai_core.config_loader import ConfigurationLoader

loader = ConfigurationLoader()
config = loader.load_from_yaml("agent_config.yaml")
agent = loader.create_agent("my_agent", config)
```

### Tracking System

Track inputs and outputs for debugging:

```python
from rakam_systems_core.ai_core.tracking import TrackingMixin

class MyAgent(TrackingMixin, BaseAgent):
    pass

agent.enable_tracking(output_dir="./tracking")
# Use agent...
agent.export_tracking_data(format='csv')
```

## Package Structure

```
rakam-systems-core/
├── src/rakam_systems_core/
│   ├── ai_core/
│   │   ├── base.py              # BaseComponent
│   │   ├── interfaces/          # Standard interfaces
│   │   ├── config_loader.py     # Configuration system
│   │   ├── tracking.py          # I/O tracking
│   │   └── mcp/                 # MCP server support
│   └── ai_utils/
│       └── logging.py           # Logging utilities
└── pyproject.toml
```

## Usage in Other Packages

### Agent Package

```python
# rakam-systems-agent uses core interfaces
from rakam_systems_core.ai_core.interfaces.agent import AgentComponent
from rakam_systems_agent import BaseAgent

agent = BaseAgent(name="my_agent", model="openai:gpt-4o")
```

### Vectorstore Package

```python
# rakam-systems-vectorstore uses core interfaces
from rakam_systems_core.ai_core.interfaces.vectorstore import VectorStore
from rakam_systems_vectorstore import ConfigurablePgVectorStore

store = ConfigurablePgVectorStore(config=config)
```

## Development

This package contains only interfaces and utilities. To contribute:

1. Install in editable mode: `pip install -e ./rakam-systems-core`
2. Make changes to interfaces or utilities
3. Ensure backward compatibility with agent and vectorstore packages
4. Update version in `pyproject.toml`

## License

Apache 2.0

<!-- ## Links

- [Main Repository](https://github.com/Rakam-AI/rakam-systems)
- [Documentation](../docs/)
- [Agent Package](../rakam-systems-agent/)
- [Vectorstore Package](../rakam-systems-vectorstore/) -->

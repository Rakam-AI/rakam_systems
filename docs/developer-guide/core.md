---
title: Core Package
---

# Core Package

The core package provides foundational abstractions used throughout the system. This package must be installed before using agent or vectorstore packages.

## BaseComponent

The base class for all components, providing lifecycle management and evaluation capabilities.

```python
from rakam_systems_core.base import BaseComponent

class BaseComponent(ABC):
    """
    Base class with:
    - name and config attributes
    - setup()/shutdown() lifecycle hooks
    - __call__ for auto-setup execution
    - Context manager support
    - Built-in evaluation harness
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.initialized = False

    def setup(self) -> None:
        """Initialize heavy resources - override in subclasses."""
        self.initialized = True

    def shutdown(self) -> None:
        """Release resources - override in subclasses."""
        self.initialized = False

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the primary operation."""
        raise NotImplementedError
```

## Interfaces

Located in `rakam_systems_core/interfaces/`, these define the contracts for various component types:

### AgentComponent

```python
from rakam_systems_core.interfaces.agent import AgentComponent, AgentInput, AgentOutput

class AgentInput:
    """Input DTO for agents."""
    input_text: str
    context: Dict[str, Any]

class AgentOutput:
    """Output DTO for agents."""
    output_text: str
    metadata: Dict[str, Any]
    output: Optional[Any]  # Structured output when output_type is used

class AgentComponent(BaseComponent, ABC):
    """Abstract agent interface with streaming and async support."""

    def run(input_data, deps=None, model_settings=None) -> AgentOutput
    async def arun(input_data, deps=None, model_settings=None) -> AgentOutput
    def stream(input_data, deps=None) -> Iterator[str]
    async def astream(input_data, deps=None) -> AsyncIterator[str]
```

### ToolComponent

```python
from rakam_systems_core.interfaces.tool import ToolComponent

class ToolComponent(BaseComponent, ABC):
    """
    Base class for callable tools, compatible with Pydantic AI.

    Attributes:
        name: Unique tool name
        description: Human-readable description
        function: The callable function
        json_schema: JSON schema for parameters
        takes_ctx: Whether tool takes context as first argument
    """

    @classmethod
    def from_function(cls, function, name, description, json_schema, takes_ctx=False):
        """Create a ToolComponent from a standalone function."""
```

### ToolRegistry

Central registry for managing tools across the system:

```python
from rakam_systems_core.interfaces.tool_registry import ToolRegistry, ToolMode

registry = ToolRegistry()

# Register a direct tool
registry.register_direct_tool(
    name="calculate",
    function=lambda x, y: x + y,
    description="Add two numbers",
    json_schema={...},
    category="math",
    tags=["arithmetic"]
)

# Register an MCP tool
registry.register_mcp_tool(
    name="search",
    mcp_server="search_server",
    mcp_tool_name="web_search",
    description="Search the web"
)

# Query tools
tools = registry.get_tools_by_category("math")
tools = registry.get_tools_by_tag("arithmetic")
tools = registry.get_tools_by_mode(ToolMode.DIRECT)
```

### LLMGateway

```python
from rakam_systems_core.interfaces.llm_gateway import LLMGateway, LLMRequest, LLMResponse

class LLMRequest(BaseModel):
    system_prompt: Optional[str]
    user_prompt: str
    temperature: Optional[float]
    max_tokens: Optional[int]
    extra_params: Dict[str, Any]

class LLMResponse(BaseModel):
    content: str
    parsed_content: Optional[Any]
    usage: Optional[Dict[str, Any]]
    model: Optional[str]
    finish_reason: Optional[str]

class LLMGateway(BaseComponent, ABC):
    """Abstract LLM gateway for provider-agnostic LLM interactions."""

    def generate(request: LLMRequest) -> LLMResponse
    def generate_structured(request: LLMRequest, schema: Type[T]) -> T
    def stream(request: LLMRequest) -> Iterator[str]
    def count_tokens(text: str, model: str = None) -> int
```

### VectorStore

```python
from rakam_systems_core.interfaces.vectorstore import VectorStore

class VectorStore(BaseComponent, ABC):
    """Abstract vector store interface."""

    def add(vectors: List[List[float]], metadatas: List[Dict]) -> Any
    def query(vector: List[float], top_k: int = 5) -> List[Dict]
    def count() -> Optional[int]
```

### Loader

```python
from rakam_systems_core.interfaces.loader import Loader

class Loader(BaseComponent, ABC):
    """Abstract document loader interface."""

    def load_as_text(source: Union[str, Path]) -> str
    def load_as_chunks(source: Union[str, Path]) -> List[str]
    def load_as_nodes(source, source_id=None, custom_metadata=None) -> List[Node]
    def load_as_vsfile(file_path, custom_metadata=None) -> VSFile
```

## Tracking System

Built-in input/output tracking for debugging and evaluation:

```python
from rakam_systems_core.tracking import TrackingManager, track_method, TrackingMixin

class MyAgent(TrackingMixin, BaseAgent):
    @track_method()
    async def arun(self, input_data, deps=None):
        return await super().arun(input_data, deps)

# Enable tracking
agent.enable_tracking(output_dir="./tracking")

# Export tracking data
agent.export_tracking_data(format='csv')
agent.export_tracking_data(format='json')

# Get statistics
stats = agent.get_tracking_statistics()
```

## Configuration Loader

Load agent configurations from YAML files:

```python
from rakam_systems_core.config_loader import ConfigurationLoader

loader = ConfigurationLoader()
config = loader.load_from_yaml("agent_config.yaml")

# Create agents from config
agent = loader.create_agent("my_agent", config)
all_agents = loader.create_all_agents(config)

# Get tool registry
registry = loader.get_tool_registry(config)

# Validate configuration
is_valid, errors = loader.validate_config("config.yaml")
```

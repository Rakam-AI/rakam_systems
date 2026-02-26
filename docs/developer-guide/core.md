---
title: Core concepts
---

# Core concepts

The core package (`rakam_systems_core`) provides foundational abstractions used throughout the system. Install it before using agent or vectorstore packages.

## BaseComponent lifecycle

All components extend `BaseComponent`, which provides lifecycle management and evaluation capabilities.

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

Use context managers or explicit `setup()`/`shutdown()` calls for proper resource management.

## Interfaces

Located in `rakam_systems_core/interfaces/`, these define the contracts for component types. Each interface extends `BaseComponent` and adds domain-specific methods.

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

## Configuration-first design

Rakam Systems embraces a configuration-first approach: modify agent behavior, vector store settings, and system parameters without touching application code.

- **Rapid iteration**: Test different models, prompts, or parameters instantly
- **Environment management**: Use different configs for dev/staging/production
- **A/B testing**: Compare performance of different settings by swapping configs
- **Team collaboration**: Non-developers can tune prompts and parameters

See [Configure with YAML](./configuration.md) for full configuration reference.

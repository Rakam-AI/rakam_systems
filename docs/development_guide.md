# Rakam Systems Development Guide

This guide provides comprehensive instructions for developers who want to contribute to or extend the `rakam_systems` framework. It covers architecture patterns, component development, testing strategies, and best practices.

## 📑 Table of Contents

1. [Getting Started](#getting-started)
2. [Architecture Overview](#architecture-overview)
3. [Core Concepts](#core-concepts)
4. [Developing Components](#developing-components)
5. [Creating Custom Agents](#creating-custom-agents)
6. [Building Tools](#building-tools)
7. [Implementing Vector Stores](#implementing-vector-stores)
8. [Creating Document Loaders](#creating-document-loaders)
9. [Configuration System](#configuration-system)
10. [Testing Guidelines](#testing-guidelines)
11. [Code Style & Conventions](#code-style--conventions)
12. [Publishing & Distribution](#publishing--distribution)

---

## Getting Started

### Development Environment Setup

```bash
# Clone the repository
git clone <repository_url>
cd rakam_systems/app/rakam_systems

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all dependencies including dev tools
pip install -e ".[complete]"

# Setup pre-commit hooks
pre-commit install

# Verify installation
python -c "from rakam_systems.ai_core.base import BaseComponent; print('✅ Setup complete!')"
```

### Project Structure

```
rakam_systems/
├── rakam_systems/               # Main package
│   ├── ai_core/                 # Core abstractions & interfaces
│   │   ├── base.py              # BaseComponent class
│   │   ├── interfaces/          # Abstract interfaces
│   │   ├── config_loader.py     # Configuration system
│   │   ├── tracking.py          # Input/output tracking
│   │   └── mcp/                 # MCP server support
│   ├── ai_agents/               # Agent implementations
│   │   ├── components/          # Agent components
│   │   │   ├── base_agent.py    # BaseAgent implementation
│   │   │   ├── llm_gateway/     # LLM provider gateways
│   │   │   ├── chat_history/    # Chat history backends
│   │   │   └── tools/           # Built-in tools
│   │   └── server/              # MCP server for agents
│   ├── ai_vectorstore/          # Vector store implementations
│   │   ├── core.py              # Node, VSFile data structures
│   │   ├── config.py            # VectorStoreConfig
│   │   └── components/          # Vector store components
│   │       ├── vectorstore/     # Store implementations
│   │       ├── embedding_model/ # Embedding models
│   │       ├── loader/          # Document loaders
│   │       └── chunker/         # Text chunkers
│   ├── ai_utils/                # Utilities (logging, metrics)
│   └── examples/                # Usage examples
├── tests/                       # Test suite
├── docs/                        # Documentation
├── pyproject.toml               # Package configuration
└── requirements.txt             # Locked dependencies
```

---

## Architecture Overview

### Design Principles

1. **Component-Based Architecture**: All functionality is encapsulated in components that extend `BaseComponent`
2. **Interface-Driven Development**: Abstract interfaces define contracts; implementations are swappable
3. **Configuration-First**: YAML/JSON configuration support for all components
4. **Provider-Agnostic**: Support multiple backends (LLMs, embeddings, vector stores)
5. **Lifecycle Management**: Components have `setup()` and `shutdown()` hooks for resource management

### Component Hierarchy

```
BaseComponent (ai_core/base.py)
├── AgentComponent (ai_core/interfaces/agent.py)
│   └── BaseAgent (ai_agents/components/base_agent.py)
├── ToolComponent (ai_core/interfaces/tool.py)
│   └── FunctionToolComponent
├── LLMGateway (ai_core/interfaces/llm_gateway.py)
│   ├── OpenAIGateway
│   └── MistralGateway
├── VectorStore (ai_core/interfaces/vectorstore.py)
│   ├── ConfigurablePgVectorStore
│   └── FAISSVectorStore
├── EmbeddingModel (ai_core/interfaces/embedding_model.py)
│   └── ConfigurableEmbeddings
├── Loader (ai_core/interfaces/loader.py)
│   ├── AdaptiveLoader
│   ├── PdfLoader, DocLoader, etc.
└── Chunker (ai_core/interfaces/chunker.py)
    └── TextChunker
```

---

## Core Concepts

### BaseComponent

Every component in the system extends `BaseComponent`, which provides:

- **Lifecycle management**: `setup()` and `shutdown()` methods
- **Auto-initialization**: `__call__` automatically calls `setup()` if needed
- **Context manager support**: Use `with` statement for automatic cleanup
- **Evaluation harness**: Built-in method for testing components

```python
from abc import abstractmethod
from typing import Any, Dict, Optional
from rakam_systems.ai_core.base import BaseComponent

class MyComponent(BaseComponent):
    """Example custom component."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.heavy_resource = None
    
    def setup(self) -> None:
        """Initialize heavy resources."""
        super().setup()  # Sets self.initialized = True
        # Initialize expensive resources here
        self.heavy_resource = self._load_model()
    
    def shutdown(self) -> None:
        """Release resources."""
        if self.heavy_resource:
            self.heavy_resource = None
        super().shutdown()  # Sets self.initialized = False
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the primary operation."""
        raise NotImplementedError
    
    def _load_model(self):
        """Private method to load model."""
        return "loaded_model"
```

### Usage Patterns

```python
# Pattern 1: Manual lifecycle
component = MyComponent("my_component")
component.setup()
result = component.run(input_data)
component.shutdown()

# Pattern 2: Auto-setup via __call__
component = MyComponent("my_component")
result = component(input_data)  # setup() called automatically

# Pattern 3: Context manager (recommended)
with MyComponent("my_component") as component:
    result = component.run(input_data)
# shutdown() called automatically
```

### Interfaces

Interfaces define contracts that implementations must follow. They are located in `ai_core/interfaces/`:

| Interface | Purpose | Key Methods |
|-----------|---------|-------------|
| `AgentComponent` | AI agents | `run()`, `arun()`, `stream()`, `astream()` |
| `ToolComponent` | Callable tools | `run()`, `acall()` |
| `LLMGateway` | LLM providers | `generate()`, `stream()`, `count_tokens()` |
| `VectorStore` | Vector storage | `add()`, `query()`, `count()` |
| `EmbeddingModel` | Text embeddings | `run()`, `encode_query()`, `encode_documents()` |
| `Loader` | Document loading | `load_as_text()`, `load_as_chunks()`, `load_as_nodes()` |
| `Chunker` | Text chunking | `run()`, `chunk_text()` |

---

## Developing Components

### Step 1: Choose the Right Interface

Determine which interface your component should implement:

```python
# For a new LLM provider
from rakam_systems.ai_core.interfaces.llm_gateway import LLMGateway

# For a new vector store backend
from rakam_systems.ai_core.interfaces.vectorstore import VectorStore

# For a new document loader
from rakam_systems.ai_core.interfaces.loader import Loader

# For a custom tool
from rakam_systems.ai_core.interfaces.tool import ToolComponent
```

### Step 2: Implement Required Methods

```python
from typing import List, Dict, Any, Optional
from rakam_systems.ai_core.interfaces.vectorstore import VectorStore

class MyCustomVectorStore(VectorStore):
    """Custom vector store implementation."""
    
    def __init__(
        self,
        name: str = "my_vectorstore",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, config)
        self.index = None
        self.dimension = config.get("dimension", 384) if config else 384
    
    def setup(self) -> None:
        """Initialize the vector index."""
        super().setup()
        # Initialize your backend here
        self.index = self._create_index()
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if self.index:
            self._save_index()
            self.index = None
        super().shutdown()
    
    def run(self, *args, **kwargs) -> Any:
        """Default run delegates to query."""
        return self.query(*args, **kwargs)
    
    def add(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> Any:
        """Add vectors to the store."""
        if not self.initialized:
            self.setup()
        # Implementation here
        return {"added": len(vectors)}
    
    def query(
        self,
        vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query similar vectors."""
        if not self.initialized:
            self.setup()
        # Implementation here
        return [{"id": i, "score": 0.9} for i in range(top_k)]
    
    def count(self) -> Optional[int]:
        """Return total vector count."""
        return len(self.index) if self.index else 0
    
    def _create_index(self):
        """Private method to create index."""
        return {}
    
    def _save_index(self):
        """Private method to save index."""
        pass
```

### Step 3: Add Configuration Support

Support both programmatic and YAML/JSON configuration:

```python
from pydantic import BaseModel, Field
from typing import Optional
import yaml

class MyVectorStoreConfig(BaseModel):
    """Configuration for MyCustomVectorStore."""
    
    name: str = "my_vectorstore"
    dimension: int = Field(default=384, ge=1)
    backend: str = Field(default="memory", pattern="^(memory|disk|redis)$")
    cache_size: int = Field(default=1000, ge=0)
    
    @classmethod
    def from_yaml(cls, path: str) -> "MyVectorStoreConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class MyCustomVectorStore(VectorStore):
    """Custom vector store with config support."""
    
    def __init__(
        self,
        config: Optional[MyVectorStoreConfig | Dict | str] = None
    ):
        # Handle different config types
        if isinstance(config, str):
            config = MyVectorStoreConfig.from_yaml(config)
        elif isinstance(config, dict):
            config = MyVectorStoreConfig(**config)
        elif config is None:
            config = MyVectorStoreConfig()
        
        super().__init__(config.name, config.to_dict())
        self.config_obj = config
```

---

## Creating Custom Agents

### Basic Agent

```python
import asyncio
from typing import Optional, List, Any
from pydantic import BaseModel
from rakam_systems.ai_agents.components import BaseAgent
from rakam_systems.ai_core.interfaces.agent import AgentInput, AgentOutput
from rakam_systems.ai_core.interfaces.tool import ToolComponent

class MyCustomAgent(BaseAgent):
    """Custom agent with specialized behavior."""
    
    def __init__(
        self,
        name: str = "custom_agent",
        model: str = "openai:gpt-4o",
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[ToolComponent]] = None,
        output_type: Optional[type[BaseModel]] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            output_type=output_type,
            **kwargs
        )
    
    async def arun(
        self,
        input_data: str | AgentInput,
        deps: Any = None,
        **kwargs
    ) -> AgentOutput:
        """Override to add custom preprocessing/postprocessing."""
        # Preprocess
        if isinstance(input_data, str):
            input_data = self._preprocess(input_data)
        
        # Call parent implementation
        result = await super().arun(input_data, deps, **kwargs)
        
        # Postprocess
        result = self._postprocess(result)
        
        return result
    
    def _preprocess(self, text: str) -> str:
        """Add preprocessing logic."""
        # Example: Add context or modify input
        return f"[User Query] {text}"
    
    def _postprocess(self, result: AgentOutput) -> AgentOutput:
        """Add postprocessing logic."""
        # Example: Format output
        return result
```

### Agent with Structured Output

```python
from pydantic import BaseModel, Field
from typing import List

class AnalysisResult(BaseModel):
    """Structured output for analysis agent."""
    
    summary: str = Field(description="Brief summary of the analysis")
    key_points: List[str] = Field(description="Key points extracted")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    needs_review: bool = Field(default=False, description="Whether human review is needed")

# Create agent with structured output
analysis_agent = BaseAgent(
    name="analysis_agent",
    model="openai:gpt-4o",
    system_prompt="You analyze text and provide structured analysis.",
    output_type=AnalysisResult  # Enforces structured output
)

# Usage
async def analyze(text: str) -> AnalysisResult:
    result = await analysis_agent.arun(f"Analyze this: {text}")
    return result.output  # Typed as AnalysisResult
```

### Agent with Dynamic System Prompts

```python
from datetime import date
from pydantic_ai import RunContext

agent = BaseAgent(
    name="dynamic_agent",
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Add dynamic prompts that are evaluated at runtime
@agent.dynamic_system_prompt
def add_date_context() -> str:
    """Add current date to system prompt."""
    return f"Today's date is {date.today().strftime('%B %d, %Y')}."

@agent.dynamic_system_prompt
def add_user_context(ctx: RunContext[dict]) -> str:
    """Add user-specific context from dependencies."""
    if ctx.deps and "user_name" in ctx.deps:
        return f"You are assisting {ctx.deps['user_name']}."
    return ""

# Usage with dependencies
result = await agent.arun(
    "What day is it?",
    deps={"user_name": "Alice"}
)
```

---

## Implementing Chat History Backends

### Custom Chat History Component

Create a custom chat history backend by implementing the `ChatHistoryComponent` interface:

```python
from typing import List, Dict, Any, Optional
from rakam_systems.ai_core.interfaces.chat_history import ChatHistoryComponent
from pydantic_ai.messages import ModelMessage

class RedisChatHistory(ChatHistoryComponent):
    """Redis-backed chat history implementation."""
    
    def __init__(
        self,
        name: str = "redis_chat_history",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, config or {})
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 6379)
        self.db = self.config.get("db", 0)
        self.client = None
    
    def setup(self) -> None:
        """Initialize Redis connection."""
        super().setup()
        import redis
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True
        )
    
    def shutdown(self) -> None:
        """Close Redis connection."""
        if self.client:
            self.client.close()
            self.client = None
        super().shutdown()
    
    def add_message(
        self,
        chat_id: str,
        message: Dict[str, Any]
    ) -> None:
        """Add a message to chat history."""
        if not self.initialized:
            self.setup()
        
        import json
        key = f"chat:{chat_id}"
        self.client.rpush(key, json.dumps(message))
    
    def get_chat_history(
        self,
        chat_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get chat history for a chat ID."""
        if not self.initialized:
            self.setup()
        
        import json
        key = f"chat:{chat_id}"
        
        if limit:
            messages = self.client.lrange(key, -limit, -1)
        else:
            messages = self.client.lrange(key, 0, -1)
        
        return [json.loads(msg) for msg in messages]
    
    def save_messages(
        self,
        chat_id: str,
        messages: List[ModelMessage]
    ) -> None:
        """Save Pydantic AI messages to history."""
        for msg in messages:
            self.add_message(chat_id, {
                "role": msg.role,
                "content": str(msg.content)
            })
    
    def get_message_history(
        self,
        chat_id: str,
        limit: Optional[int] = None
    ) -> List[ModelMessage]:
        """Get history as Pydantic AI ModelMessage objects."""
        from pydantic_ai.messages import (
            ModelMessage,
            UserPromptPart,
            TextPart
        )
        
        messages = self.get_chat_history(chat_id, limit)
        result = []
        
        for msg in messages:
            if msg["role"] == "user":
                result.append(ModelMessage(
                    role="user",
                    parts=[UserPromptPart(content=msg["content"])]
                ))
            elif msg["role"] == "assistant":
                result.append(ModelMessage(
                    role="assistant",
                    parts=[TextPart(content=msg["content"])]
                ))
        
        return result
    
    def delete_chat_history(self, chat_id: str) -> None:
        """Delete chat history for a chat ID."""
        if not self.initialized:
            self.setup()
        
        key = f"chat:{chat_id}"
        self.client.delete(key)
    
    def get_all_chat_ids(self) -> List[str]:
        """Get all chat IDs."""
        if not self.initialized:
            self.setup()
        
        keys = self.client.keys("chat:*")
        return [key.replace("chat:", "") for key in keys]
    
    def clear_all(self) -> None:
        """Clear all chat histories."""
        if not self.initialized:
            self.setup()
        
        keys = self.client.keys("chat:*")
        if keys:
            self.client.delete(*keys)
```

### Using Custom Chat History

```python
import asyncio
from rakam_systems.ai_agents import BaseAgent

async def main():
    # Initialize chat history
    history = RedisChatHistory(config={
        "host": "localhost",
        "port": 6379
    })
    
    # Create agent
    agent = BaseAgent(
        name="chat_agent",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant."
    )
    
    # Use with chat history
    chat_id = "user_123"
    
    # Get existing history
    message_history = history.get_message_history(chat_id)
    
    # Run agent with history
    result = await agent.arun(
        "What did we talk about?",
        message_history=message_history
    )
    
    # Save new messages
    history.save_messages(chat_id, result.all_messages())
    
    print(result.output_text)

asyncio.run(main())
```

---

## Building Tools

### Method 1: Class-Based Tool

```python
from typing import Any, Dict
from rakam_systems.ai_core.interfaces.tool import ToolComponent

class WeatherTool(ToolComponent):
    """Tool to get weather information."""
    
    def __init__(self, api_key: str = None):
        super().__init__(
            name="get_weather",
            description="Get current weather for a location",
            json_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["city"],
                "additionalProperties": False
            }
        )
        self.api_key = api_key
    
    def run(self, city: str, units: str = "celsius") -> str:
        """Fetch weather data."""
        # Implementation here
        return f"Weather in {city}: 22°{units[0].upper()}, Sunny"
```

### Method 2: Function-Based Tool (Recommended for Simple Tools)

```python
from rakam_systems.ai_core.interfaces.tool import ToolComponent

def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Calculate BMI given weight and height."""
    bmi = weight_kg / (height_m ** 2)
    category = (
        "Underweight" if bmi < 18.5 else
        "Normal" if bmi < 25 else
        "Overweight" if bmi < 30 else
        "Obese"
    )
    return {"bmi": round(bmi, 2), "category": category}

# Create tool from function
bmi_tool = ToolComponent.from_function(
    function=calculate_bmi,
    name="calculate_bmi",
    description="Calculate Body Mass Index from weight and height",
    json_schema={
        "type": "object",
        "properties": {
            "weight_kg": {
                "type": "number",
                "description": "Weight in kilograms"
            },
            "height_m": {
                "type": "number", 
                "description": "Height in meters"
            }
        },
        "required": ["weight_kg", "height_m"],
        "additionalProperties": False
    }
)
```

### Method 3: Async Tool

```python
import aiohttp
from rakam_systems.ai_core.interfaces.tool import ToolComponent

class AsyncSearchTool(ToolComponent):
    """Async tool for web search."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information",
            json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        )
    
    async def run(self, query: str) -> str:
        """Perform async web search."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.search.com?q={query}") as resp:
                data = await resp.json()
                return data.get("results", [])
    
    # acall is automatically handled for async run methods
```

### Registering Tools with ToolRegistry

```python
from rakam_systems.ai_core.interfaces.tool_registry import ToolRegistry, ToolMode

registry = ToolRegistry()

# Register tools
registry.register_direct_tool(
    name="weather",
    function=get_weather_fn,
    description="Get weather information",
    json_schema={...},
    category="utility",
    tags=["weather", "external"]
)

registry.register_direct_tool(
    name="calculator",
    function=calculate_fn,
    description="Perform calculations",
    json_schema={...},
    category="math",
    tags=["calculation"]
)

# Query tools
math_tools = registry.get_tools_by_category("math")
external_tools = registry.get_tools_by_tag("external")
all_direct_tools = registry.get_tools_by_mode(ToolMode.DIRECT)
```

---

## Implementing Vector Stores

### Custom Vector Store

```python
from typing import List, Dict, Any, Optional
from rakam_systems.ai_core.interfaces.vectorstore import VectorStore
from rakam_systems.ai_vectorstore.core import Node, NodeMetadata

class RedisVectorStore(VectorStore):
    """Redis-backed vector store implementation."""
    
    def __init__(
        self,
        name: str = "redis_vectorstore",
        host: str = "localhost",
        port: int = 6379,
        index_name: str = "vectors",
        dimension: int = 384,
        config: Optional[Dict] = None
    ):
        super().__init__(name, config or {})
        self.host = host
        self.port = port
        self.index_name = index_name
        self.dimension = dimension
        self.client = None
    
    def setup(self) -> None:
        """Connect to Redis and create index."""
        super().setup()
        import redis
        from redis.commands.search.field import VectorField, TextField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        
        self.client = redis.Redis(host=self.host, port=self.port)
        
        # Create vector index
        try:
            self.client.ft(self.index_name).create_index(
                fields=[
                    VectorField("embedding", "HNSW", {
                        "TYPE": "FLOAT32",
                        "DIM": self.dimension,
                        "DISTANCE_METRIC": "COSINE"
                    }),
                    TextField("content"),
                ],
                definition=IndexDefinition(
                    prefix=[f"{self.index_name}:"],
                    index_type=IndexType.HASH
                )
            )
        except Exception:
            pass  # Index already exists
    
    def shutdown(self) -> None:
        """Close Redis connection."""
        if self.client:
            self.client.close()
            self.client = None
        super().shutdown()
    
    def run(self, *args, **kwargs) -> Any:
        return self.query(*args, **kwargs)
    
    def add(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add vectors to Redis."""
        if not self.initialized:
            self.setup()
        
        import numpy as np
        
        added = 0
        for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
            doc_id = metadata.get("id", f"doc_{i}")
            self.client.hset(
                f"{self.index_name}:{doc_id}",
                mapping={
                    "embedding": np.array(vector, dtype=np.float32).tobytes(),
                    "content": metadata.get("content", ""),
                    **metadata
                }
            )
            added += 1
        
        return {"added": added}
    
    def query(
        self,
        vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self.initialized:
            self.setup()
        
        import numpy as np
        from redis.commands.search.query import Query
        
        query_vector = np.array(vector, dtype=np.float32).tobytes()
        
        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("content", "score")
            .dialect(2)
        )
        
        results = self.client.ft(self.index_name).search(
            q, query_params={"vec": query_vector}
        )
        
        return [
            {
                "id": doc.id,
                "content": doc.content,
                "score": float(doc.score)
            }
            for doc in results.docs
        ]
    
    def count(self) -> Optional[int]:
        """Count total documents."""
        if not self.client:
            return 0
        info = self.client.ft(self.index_name).info()
        return int(info.get("num_docs", 0))
```

---

## Creating Document Loaders

### Custom Loader

```python
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
from rakam_systems.ai_core.interfaces.loader import Loader
from rakam_systems.ai_vectorstore.core import Node, NodeMetadata, VSFile

class XMLLoader(Loader):
    """Loader for XML documents."""
    
    def __init__(
        self,
        name: str = "xml_loader",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, config or {})
        self.chunk_size = self.config.get("chunk_size", 512)
        self.encoding = self.config.get("encoding", "utf-8")
    
    def run(self, source: Union[str, Path]) -> List[str]:
        """Default run delegates to load_as_chunks."""
        return self.load_as_chunks(source)
    
    def load_as_text(self, source: Union[str, Path]) -> str:
        """Load XML as plain text (elements removed)."""
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(source)
        root = tree.getroot()
        
        # Extract all text content
        texts = []
        for elem in root.iter():
            if elem.text:
                texts.append(elem.text.strip())
            if elem.tail:
                texts.append(elem.tail.strip())
        
        return " ".join(filter(None, texts))
    
    def load_as_chunks(self, source: Union[str, Path]) -> List[str]:
        """Load XML and split into chunks."""
        text = self.load_as_text(source)
        
        # Simple chunking by character count
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """Load XML as nodes with metadata."""
        source_path = Path(source)
        chunks = self.load_as_chunks(source)
        
        nodes = []
        for i, chunk in enumerate(chunks):
            metadata = NodeMetadata(
                source_file_uuid=source_id or str(source_path),
                position=i,
                custom=custom_metadata or {}
            )
            nodes.append(Node(content=chunk, metadata=metadata))
        
        return nodes
    
    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VSFile:
        """Load XML as VSFile with nodes attached."""
        vsfile = VSFile(file_path=str(file_path))
        vsfile.nodes = self.load_as_nodes(
            file_path,
            source_id=str(vsfile.uuid),
            custom_metadata=custom_metadata
        )
        return vsfile
```

### Loader with Image Extraction

Modern loaders support image extraction. Here's how to implement it:

```python
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
from rakam_systems.ai_core.interfaces.loader import Loader
from rakam_systems.ai_vectorstore.core import Node, NodeMetadata, VSFile

class ImageAwareLoader(Loader):
    """Loader with image extraction support."""
    
    def __init__(
        self,
        name: str = "image_loader",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, config or {})
        self.extract_images = self.config.get("extract_images", False)
        self.image_path = self.config.get("image_path", "./extracted_images")
        self.write_images = self.config.get("write_images", True)
        self._image_paths = {}  # Store image ID to path mapping
        
        if self.extract_images and self.write_images:
            Path(self.image_path).mkdir(parents=True, exist_ok=True)
    
    def load_as_text(self, source: Union[str, Path]) -> str:
        """Load document as text with image references."""
        # Extract text and images
        text_content = self._extract_text(source)
        
        if self.extract_images:
            images = self._extract_images(source)
            for img_id, img_data in images.items():
                if self.write_images:
                    img_path = self._save_image(img_id, img_data)
                    self._image_paths[img_id] = img_path
                # Add image reference to text
                text_content += f"\n[Image: {img_id}]"
        
        return text_content
    
    def get_image_paths(self) -> Dict[str, str]:
        """Get mapping of image IDs to absolute paths."""
        return {
            img_id: str(Path(path).absolute())
            for img_id, path in self._image_paths.items()
        }
    
    def _extract_text(self, source: Union[str, Path]) -> str:
        """Extract text from document."""
        # Implementation specific to document type
        pass
    
    def _extract_images(self, source: Union[str, Path]) -> Dict[str, bytes]:
        """Extract images from document."""
        # Implementation specific to document type
        # Returns dict of {image_id: image_bytes}
        pass
    
    def _save_image(self, img_id: str, img_data: bytes) -> str:
        """Save image to disk and return path."""
        img_path = Path(self.image_path) / f"{img_id}.png"
        img_path.write_bytes(img_data)
        return str(img_path)
```

**Example Usage:**

```python
loader = ImageAwareLoader(config={
    "extract_images": True,
    "image_path": "./my_images",
    "write_images": True
})

# Load document
text = loader.load_as_text("document.pdf")

# Get extracted image paths
image_paths = loader.get_image_paths()
for img_id, img_path in image_paths.items():
    print(f"Extracted image {img_id} to {img_path}")
```

### Integrating with AdaptiveLoader

To make your loader work with `AdaptiveLoader`, register the file extension:

```python
from rakam_systems.ai_vectorstore.components.loader import AdaptiveLoader

class EnhancedAdaptiveLoader(AdaptiveLoader):
    """AdaptiveLoader with XML support."""
    
    EXTENSION_LOADERS = {
        **AdaptiveLoader.EXTENSION_LOADERS,
        ".xml": XMLLoader,
        ".xhtml": XMLLoader,
    }
```

---

## Configuration System

### YAML Configuration

The framework supports comprehensive YAML configuration:

```yaml
# config/agent_config.yaml
version: "1.0"

# Prompt library
prompts:
  assistant:
    name: "assistant"
    system_prompt: |
      You are a helpful AI assistant.
      Always be accurate and helpful.
    skills:
      - "Information retrieval"
      - "Task automation"

# Tool library
tools:
  search:
    name: "search"
    type: "direct"
    module: "myapp.tools"
    function: "search_documents"
    description: "Search documents"
    json_schema:
      type: "object"
      properties:
        query:
          type: "string"
      required: ["query"]

# Agent configurations
agents:
  main_agent:
    name: "main_agent"
    model_config:
      model: "openai:gpt-4o"
      temperature: 0.7
      max_tokens: 2000
    prompt_config: "assistant"
    tools:
      - "search"
    enable_tracking: true
```

### Loading Configuration

```python
from rakam_systems.ai_core.config_loader import ConfigurationLoader

loader = ConfigurationLoader()

# Load from YAML
config = loader.load_from_yaml("config/agent_config.yaml")

# Create single agent
agent = loader.create_agent("main_agent", config)

# Create all agents
all_agents = loader.create_all_agents(config)

# Get tool registry
registry = loader.get_tool_registry(config)

# Validate configuration
is_valid, errors = loader.validate_config("config/agent_config.yaml")
if not is_valid:
    for error in errors:
        print(f"Config error: {error}")
```

---

## Testing Guidelines

### Unit Test Structure

```python
# tests/test_my_component.py
import pytest
from typing import List, Dict, Any
from rakam_systems.ai_core.interfaces.vectorstore import VectorStore

class DummyVectorStore(VectorStore):
    """Mock implementation for testing."""
    
    def __init__(self):
        super().__init__("dummy_store")
        self.data = []
    
    def run(self, *args, **kwargs):
        return self.query(*args, **kwargs)
    
    def add(self, vectors: List[List[float]], metadatas: List[Dict]) -> Dict:
        self.data.extend(zip(vectors, metadatas))
        return {"added": len(vectors)}
    
    def query(self, vector: List[float], top_k: int = 5) -> List[Dict]:
        return [{"id": i, "score": 0.9} for i in range(min(top_k, len(self.data)))]
    
    def count(self) -> int:
        return len(self.data)


class TestVectorStore:
    """Test suite for VectorStore implementations."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh store for each test."""
        store = DummyVectorStore()
        yield store
        store.shutdown()
    
    def test_add_vectors(self, store):
        """Test adding vectors to store."""
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"id": 1}, {"id": 2}]
        
        result = store.add(vectors, metadatas)
        
        assert result["added"] == 2
        assert store.count() == 2
    
    def test_query_vectors(self, store):
        """Test querying vectors."""
        store.add([[0.1, 0.2]], [{"id": 1}])
        
        results = store.query([0.1, 0.2], top_k=5)
        
        assert len(results) <= 5
        assert all("score" in r for r in results)
    
    def test_lifecycle(self, store):
        """Test setup/shutdown lifecycle."""
        assert not store.initialized
        
        store.setup()
        assert store.initialized
        
        store.shutdown()
        assert not store.initialized
    
    def test_evaluate_harness(self, store):
        """Test built-in evaluation harness."""
        store.add([[0.1]], [{"id": 1}])
        
        test_cases = {
            "query": [
                {"args": [[0.1, 0.2]], "kwargs": {"top_k": 3}}
            ]
        }
        
        results = store.evaluate(
            methods=["query"],
            test_cases=test_cases,
            verbose=False
        )
        
        assert "query" in results
        assert results["query"][0]["success"]
```

### Async Test Structure

```python
import pytest
import asyncio

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_async_agent():
    """Test async agent methods."""
    from rakam_systems.ai_agents.components import BaseAgent
    
    agent = BaseAgent(
        name="test_agent",
        model="openai:gpt-4o-mini",
        system_prompt="You are a test assistant."
    )
    
    result = await agent.arun("Say 'hello'")
    
    assert result.output_text is not None
    assert len(result.output_text) > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rakam_systems --cov-report=html

# Run specific test file
pytest tests/test_my_component.py

# Run tests matching pattern
pytest -k "test_vector"

# Run async tests
pytest -v tests/test_async_*.py
```

---

## Code Style & Conventions

### Python Style

Follow PEP 8 with these project-specific conventions:

```python
# Imports: Standard library, third-party, local (separated by blank lines)
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from rakam_systems.ai_core.base import BaseComponent


# Class docstrings: Brief description + Attributes + Methods
class MyComponent(BaseComponent):
    """Brief description of the component.
    
    This component does X, Y, and Z. Use it when you need to...
    
    Attributes:
        name: Component identifier
        config: Configuration dictionary
        some_param: Description of parameter
    
    Example:
        ```python
        component = MyComponent("my_comp", config={"key": "value"})
        with component:
            result = component.run(input_data)
        ```
    """


# Method docstrings: Brief description + Args + Returns + Raises
def my_method(self, param1: str, param2: int = 10) -> Dict[str, Any]:
    """Brief description of what this method does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Dictionary containing:
            - key1: Description
            - key2: Description
    
    Raises:
        ValueError: When param1 is empty
        RuntimeError: When operation fails
    """
```

### Type Hints

Always use type hints:

```python
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

T = TypeVar("T")

class GenericComponent(BaseComponent, Generic[T]):
    def run(self, data: T) -> T:
        ...

def process_items(
    items: List[Dict[str, Any]],
    filter_fn: Optional[callable] = None,
    *,
    limit: int = 100
) -> List[Dict[str, Any]]:
    ...
```

### Formatting Tools

```bash
# Format code with black
black rakam_systems/

# Lint with ruff
ruff check rakam_systems/

# Type check (optional)
mypy rakam_systems/
```

---

## Publishing & Distribution

### Version Management

Update version in `pyproject.toml`:

```toml
[project]
name = "rakam-systems"
version = "0.2.4"  # Update this
```

### Building the Package

```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# Check distribution
twine check dist/*
```

### Publishing to PyPI

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ rakam-systems

# Upload to PyPI
twine upload dist/*
```

### Creating a Release

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit changes: `git commit -m "Release v0.2.4"`
4. Tag release: `git tag v0.2.4`
5. Push: `git push && git push --tags`
6. Build and publish to PyPI

---

## Additional Resources

- **Component Documentation**: `docs/components.md`
- **Installation Guide**: `INSTALLATION.md`
- **Example Configurations**: `rakam_systems/examples/configs/`
- **Agent Examples**: `rakam_systems/examples/ai_agents_examples/`
- **Vector Store Examples**: `rakam_systems/examples/ai_vectorstore_examples/`

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check component-specific READMEs
- **Examples**: Explore the `examples/` directory for working code

---

**Happy Developing! 🚀**

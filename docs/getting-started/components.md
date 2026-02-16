# Components Documentation

Rakam Systems is a modular AI framework designed to build production-ready AI applications. It provides a comprehensive set of components for building AI agents, vector stores, and LLM-powered applications.

## 📑 Table of Contents

1. [🏗️ Architecture Overview](#️-architecture-overview)
2. [🧱 Core Package (`rakam-systems-core`)](#core-package-rakam-systems-core)
3. [🤖 Agent Package (`rakam-systems-agent`)](#-agent-package-rakam-system-agent)
4. [🔍 Vectorstore Package (`rakam-systems-vectorstore`)](#-vectorstore-package-rakam-system-vectorstore)
5. [⚙️ Configuration System](#️-configuration-system)
6. [🚀 Quick Start Examples](#-quick-start-examples)
7. [🌍 Environment Variables](#environment-variables)
8. [✅ Best Practices](#-best-practices)
9. [📚 Further Reading](#-further-reading)

---

## 🏗️ Architecture Overview

Rakam Systems is organized into three independent packages:

```
rakam-systems-inhouse/
├── rakam-system-core/           # Core abstractions, interfaces, and base classes
│   └── src/rakam_system_core/
│       ├── ai_core/             # Core interfaces and base component
│       │   ├── base.py          # BaseComponent
│       │   ├── interfaces/      # Abstract interfaces
│       │   ├── config_loader.py # Configuration system
│       │   └── tracking.py      # Input/output tracking
│       └── ai_utils/            # Logging utilities
├── rakam-system-agent/          # Agent implementations (depends on core)
│   └── src/rakam_systems_agent/
│       └── components/
│           ├── base_agent.py    # BaseAgent implementation
│           ├── llm_gateway/     # LLM provider gateways
│           ├── chat_history/    # Chat history backends
│           └── tools/           # Built-in tools
└── rakam-system-vectorstore/    # Vector storage (depends on core)
    └── src/rakam_systems_vectorstore/
        ├── core.py              # Node, VSFile data structures
        ├── config.py            # VectorStoreConfig
        └── components/
            ├── vectorstore/     # Store implementations
            ├── embedding_model/ # Embedding models
            ├── loader/          # Document loaders
            └── chunker/         # Text chunkers
```

### Design Principles

- **Modular Architecture**: Three independent packages that can be installed separately
- **Clear Dependencies**: Agent and vectorstore packages depend on core
- **Component-Based**: All components extend `BaseComponent` with lifecycle management (`setup()`, `shutdown()`)
- **Interface-Driven**: Abstract interfaces define contracts for extensibility
- **Configuration-First**: YAML/JSON configuration support for all components
- **Provider-Agnostic**: Support for multiple LLM providers, embedding models, and vector stores

---

## Core Package (`rakam-systems-core`)

The core package provides foundational abstractions used throughout the system. This package must be installed before using agent or vectorstore packages.

### BaseComponent

The base class for all components, providing lifecycle management and evaluation capabilities.

```python
from rakam_system_core.base import BaseComponent

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

### Interfaces

Located in `interfaces/`, these define the contracts for various component types:

#### AgentComponent

```python
from rakam_system_core.interfaces.agent import AgentComponent, AgentInput, AgentOutput

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

#### ToolComponent

```python
from rakam_system_core.interfaces.tool import ToolComponent

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

#### ToolRegistry

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

#### LLMGateway

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

#### VectorStore

```python
from rakam_systems_core.interfaces.vectorstore import VectorStore

class VectorStore(BaseComponent, ABC):
    """Abstract vector store interface."""

    def add(vectors: List[List[float]], metadatas: List[Dict]) -> Any
    def query(vector: List[float], top_k: int = 5) -> List[Dict]
    def count() -> Optional[int]
```

#### Loader

```python
from rakam_systems_core.interfaces.loader import Loader

class Loader(BaseComponent, ABC):
    """Abstract document loader interface."""

    def load_as_text(source: Union[str, Path]) -> str
    def load_as_chunks(source: Union[str, Path]) -> List[str]
    def load_as_nodes(source, source_id=None, custom_metadata=None) -> List[Node]
    def load_as_vsfile(file_path, custom_metadata=None) -> VSFile
```

### Tracking System

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

### Configuration Loader

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

---

## 🤖 Agent Package (`rakam-system-agent`)

The agent package provides AI agent implementations powered by Pydantic AI. Install with `pip install rakam-systems-agent` (requires core).

### BaseAgent

The main agent implementation using Pydantic AI:

```python
from rakam_systems_agent import BaseAgent
from rakam_systems_core.interfaces.agent import AgentInput, AgentOutput, ModelSettings

agent = BaseAgent(
    name="my_agent",
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
    tools=[my_tool],  # Optional tools
    output_type=MyOutputModel,  # Optional structured output
    enable_tracking=True  # Optional tracking
)

# Async inference (required for Pydantic AI)
result = await agent.arun("What is AI?")
print(result.output_text)

# With dependencies
result = await agent.arun("Hello", deps={"user_id": "123"})

# With model settings
settings = ModelSettings(temperature=0.5, max_tokens=1000)
result = await agent.arun("Explain quantum computing", model_settings=settings)

# Streaming
async for chunk in agent.astream("Tell me a story"):
    print(chunk, end="")
```

#### Dynamic System Prompts

Dynamic system prompts allow you to inject context at runtime based on current state, user information, or external data:

```python
from datetime import date, datetime
from pydantic_ai import RunContext

agent = BaseAgent(
    name="dynamic_agent",
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Method 1: Decorator syntax
@agent.dynamic_system_prompt
def add_date() -> str:
    """Add current date to system prompt."""
    return f"Today's date is {date.today().strftime('%B %d, %Y')}."

@agent.dynamic_system_prompt
def add_user_context(ctx: RunContext[dict]) -> str:
    """Add user-specific context from dependencies."""
    if ctx.deps and "user_name" in ctx.deps:
        return f"You are assisting {ctx.deps['user_name']}."
    return ""

# Method 2: Direct registration
def add_time_context() -> str:
    """Add current time to system prompt."""
    return f"Current time: {datetime.now().strftime('%H:%M:%S')}"

agent.add_dynamic_system_prompt(add_time_context)

# Method 3: Async dynamic prompts
@agent.dynamic_system_prompt
async def fetch_external_context(ctx: RunContext[dict]) -> str:
    """Fetch and add external context asynchronously."""
    # Example: fetch from API or database
    import asyncio
    await asyncio.sleep(0.1)
    return "Additional context from external source."

# Usage with dependencies
result = await agent.arun(
    "What day is it?",
    deps={"user_name": "Alice", "user_id": "123"}
)
```

### LLM Gateways

#### OpenAI Gateway

```python
from rakam_systems_agent import OpenAIGateway, LLMRequest

gateway = OpenAIGateway(
    model="gpt-4o",
    api_key="...",  # Or use OPENAI_API_KEY env var
    default_temperature=0.7
)

# Text generation
request = LLMRequest(
    system_prompt="You are a helpful assistant",
    user_prompt="What is AI?",
    temperature=0.7
)
response = gateway.generate(request)
print(response.content)

# Structured output
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

result = gateway.generate_structured(request, Answer)
print(result.answer, result.confidence)

# Streaming
for chunk in gateway.stream(request):
    print(chunk, end="")

# Token counting
token_count = gateway.count_tokens("Hello, world!")
```

#### Mistral Gateway

```python
from rakam_systems_agent import MistralGateway

gateway = MistralGateway(
    model="mistral-large-latest",
    api_key="..."  # Or use MISTRAL_API_KEY env var
)
```

#### Gateway Factory

```python
from rakam_systems_agent import LLMGatewayFactory, get_llm_gateway

# Using factory
gateway = LLMGatewayFactory.create(
    provider="openai",
    model="gpt-4o",
    api_key="..."
)

# Using convenience function
gateway = get_llm_gateway(provider="openai", model="gpt-4o")
```

### Chat History

#### JSON Chat History

```python
from rakam_systems_agent.components.chat_history import JSONChatHistory

history = JSONChatHistory(config={"storage_path": "./chat_history.json"})

# Add messages
history.add_message("chat123", {"role": "user", "content": "Hello!"})
history.add_message("chat123", {"role": "assistant", "content": "Hi there!"})

# Get history
messages = history.get_chat_history("chat123")
readable = history.get_readable_chat_history("chat123")

# Pydantic AI integration
message_history = history.get_message_history("chat123")
result = await agent.run("Hello", message_history=message_history)
history.save_messages("chat123", result.all_messages())

# Manage chats
all_chats = history.get_all_chat_ids()
history.delete_chat_history("chat123")
history.clear_all()
```

#### SQL Chat History (SQLite)

```python
from rakam_systems_agent.components.chat_history import SQLChatHistory

history = SQLChatHistory(config={"db_path": "./chat_history.db"})

# Same API as JSON Chat History
history.add_message("chat123", {"role": "user", "content": "Hello!"})
history.add_message("chat123", {"role": "assistant", "content": "Hi there!"})

# Get history
messages = history.get_chat_history("chat123")

# Pydantic AI integration
message_history = history.get_message_history("chat123")
result = await agent.run("Hello", message_history=message_history)
history.save_messages("chat123", result.all_messages())
```

#### PostgreSQL Chat History

For production deployments with PostgreSQL-backed storage:

```python
from rakam_systems_agent.components.chat_history import PostgresChatHistory

# Configuration
history = PostgresChatHistory(config={
    "host": "localhost",
    "port": 5432,
    "database": "chat_db",
    "user": "postgres",
    "password": "postgres"
})

# Or use environment variables (POSTGRES_HOST, POSTGRES_PORT, etc.)
history = PostgresChatHistory()

# Same API as other chat history backends
history.add_message("chat123", {"role": "user", "content": "Hello!"})
history.add_message("chat123", {"role": "assistant", "content": "Hi there!"})

# Get history
messages = history.get_chat_history("chat123")
readable = history.get_readable_chat_history("chat123")

# Pydantic AI integration
message_history = history.get_message_history("chat123")
result = await agent.run("Hello", message_history=message_history)
history.save_messages("chat123", result.all_messages())

# Manage chats
all_chats = history.get_all_chat_ids()
history.delete_chat_history("chat123")
history.clear_all()

# Cleanup
history.shutdown()
```

---

## 🔍 Vectorstore Package (`rakam-system-vectorstore`)

The vectorstore package provides vector database solutions and document processing. Install with `pip install -e ./rakam-system-vectorstore` (requires core).

### Core Data Structures

```python
from rakam_systems_vectorstore.core import Node, NodeMetadata, VSFile

# VSFile - Represents a document source
vsfile = VSFile(file_path="/path/to/document.pdf")
print(vsfile.uuid, vsfile.file_name, vsfile.mime_type)

# NodeMetadata - Metadata for document chunks
metadata = NodeMetadata(
    source_file_uuid=str(vsfile.uuid),
    position=0,  # Page number or chunk position
    custom={"author": "John", "date": "2024-01-01"}
)

# Node - A chunk with content and metadata
node = Node(content="Document content here...", metadata=metadata)
node.embedding = [0.1, 0.2, 0.3, ...]  # Set after embedding
```

### ConfigurablePgVectorStore

Enhanced PostgreSQL vector store with full configuration support:

```python
from rakam_systems_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

# From configuration object
config = VectorStoreConfig()
store = ConfigurablePgVectorStore(config=config)

# From YAML file
store = ConfigurablePgVectorStore(config="vectorstore_config.yaml")

# From dictionary
store = ConfigurablePgVectorStore(config={
    "name": "my_store",
    "embedding": {
        "model_type": "sentence_transformer",
        "model_name": "Snowflake/snowflake-arctic-embed-m"
    },
    "search": {
        "similarity_metric": "cosine",
        "enable_hybrid_search": True,
        "hybrid_alpha": 0.7
    }
})

# Setup (initializes embedding model, database tables)
store.setup()

# Add documents
store.add_nodes(nodes)
store.add_vsfile(vsfile)

# Vector search (semantic similarity)
results = store.search("What is machine learning?", top_k=5)

# Hybrid search (combines vector + keyword search)
results = store.hybrid_search("machine learning", top_k=10, alpha=0.7)

# Keyword search (full-text search with BM25 or ts_rank)
results = store.keyword_search(
    query="machine learning algorithms",
    top_k=10,
    ranking_algorithm="bm25",  # or "ts_rank"
    k1=1.2,  # BM25 parameter
    b=0.75   # BM25 parameter
)

# Update vectors
store.update_vector(node_id, new_embedding)

# Cleanup
store.shutdown()
```

#### Keyword Search

Full-text search using PostgreSQL's built-in capabilities with BM25 or ts_rank ranking:

```python
from rakam_systems_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

config = VectorStoreConfig(
    search={
        "keyword_ranking_algorithm": "bm25",  # or "ts_rank"
        "keyword_k1": 1.2,  # BM25 k1 parameter
        "keyword_b": 0.75   # BM25 b parameter
    }
)

store = ConfigurablePgVectorStore(config=config)
store.setup()

# Keyword search with BM25 ranking
results = store.keyword_search(
    query="machine learning neural networks",
    top_k=10,
    ranking_algorithm="bm25"
)

# Keyword search with ts_rank
results = store.keyword_search(
    query="deep learning",
    top_k=10,
    ranking_algorithm="ts_rank"
)

# Results include content and relevance scores
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:200]}...")
```

**Ranking Algorithms:**

- **BM25**: Best Match 25, probabilistic ranking function
  - `k1`: Term frequency saturation parameter (default: 1.2)
  - `b`: Length normalization parameter (default: 0.75)
- **ts_rank**: PostgreSQL's text search ranking function
  - Weights different parts of documents differently
  - Good for structured documents

#### Multi-Model Support

Each embedding model automatically gets dedicated tables:

```python
# Using different models - each gets its own tables
store_minilm = ConfigurablePgVectorStore(config=config_minilm)
store_mpnet = ConfigurablePgVectorStore(config=config_mpnet)

# Table names are based on model names:
# - application_nodeentry_all_minilm_l6_v2
# - application_nodeentry_snowflake_arctic_embed_m

# Disable model-specific tables if needed (not recommended)
store = ConfigurablePgVectorStore(
    config=config,
    use_dimension_specific_tables=False
)
```

### ConfigurableEmbeddings

Multi-backend embedding model with unified interface:

```python
from rakam_systems_vectorstore import ConfigurableEmbeddings, create_embedding_model

# Using Sentence Transformers (local)
embeddings = ConfigurableEmbeddings(config={
    "model_type": "sentence_transformer",
    "model_name": "Snowflake/snowflake-arctic-embed-m",
    "batch_size": 128,
    "normalize": True
})

# Using OpenAI (with batch processing)
embeddings = ConfigurableEmbeddings(config={
    "model_type": "openai",
    "model_name": "text-embedding-3-small",
    "api_key": "...",  # Or use OPENAI_API_KEY
    "batch_size": 100   # OpenAI supports larger batches
})

# Using Cohere
embeddings = ConfigurableEmbeddings(config={
    "model_type": "cohere",
    "model_name": "embed-english-v3.0",
    "api_key": "..."  # Or use COHERE_API_KEY
})

# Using HuggingFace models with authentication
embeddings = ConfigurableEmbeddings(config={
    "model_type": "sentence_transformer",
    "model_name": "private/model-name",
    # Uses HUGGINGFACE_TOKEN environment variable
})

embeddings.setup()

# Encode texts with automatic batch processing
vectors = embeddings.run(["Hello world", "How are you?"])

# Encode large datasets with progress tracking
large_texts = ["text" + str(i) for i in range(10000)]
vectors = embeddings.run(large_texts)  # Shows progress bar

# Encode queries (optimized for single texts)
query_vector = embeddings.encode_query("What is AI?")

# Encode documents (optimized for batches)
doc_vectors = embeddings.encode_documents(documents)

# Get dimension
dim = embeddings.embedding_dimension
```

**Performance Features:**

- Automatic batch processing with progress tracking
- Memory optimization with garbage collection
- Token truncation for oversized texts
- Mini-batch processing for large datasets
- CUDA memory management for GPU acceleration

#### Factory Function

```python
embeddings = create_embedding_model(
    model_type="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
    batch_size=64
)
```

### AdaptiveLoader

Automatically detects and processes various file types:

```python
from rakam_systems_vectorstore import AdaptiveLoader, create_adaptive_loader

loader = AdaptiveLoader(config={
    "encoding": "utf-8",
    "chunk_size": 512,
    "chunk_overlap": 50
})

# Supported file types:
# - Text: .txt, .text
# - Markdown: .md, .markdown
# - Documents: .pdf, .docx, .doc, .odt
# - Email: .eml, .msg
# - Data: .json, .csv, .tsv, .xlsx, .xls
# - HTML: .html, .htm, .xhtml
# - Code: .py, .js, .ts, .java, .cpp, .go, .rs, .rb, etc.

# Load as single text
text = loader.load_as_text("document.pdf")

# Load as chunks
chunks = loader.load_as_chunks("document.pdf")

# Load as nodes (with metadata)
nodes = loader.load_as_nodes("document.pdf", custom_metadata={"category": "science"})

# Load as VSFile
vsfile = loader.load_as_vsfile("document.pdf")

# Also handles raw text
chunks = loader.load_as_chunks("This is raw text content...")
```

#### Factory Function

```python
loader = create_adaptive_loader(
    chunk_size=1024,
    chunk_overlap=100,
    encoding='utf-8'
)
```

### Specialized Loaders

Located in `ai_vectorstore/components/loader/`:

| Loader           | File Types              | Features                                                                           |
| ---------------- | ----------------------- | ---------------------------------------------------------------------------------- |
| `PdfLoader`      | `.pdf`                  | Advanced PDF processing with Docling, image extraction, table detection            |
| `PdfLoaderLight` | `.pdf`                  | Lightweight PDF processing with pymupdf4llm, markdown conversion, image extraction |
| `DocLoader`      | `.docx`, `.doc`         | Microsoft Word documents, image extraction                                         |
| `OdtLoader`      | `.odt`                  | OpenDocument Text, image extraction                                                |
| `MdLoader`       | `.md`                   | Markdown with structure preservation, YAML frontmatter                             |
| `HtmlLoader`     | `.html`, `.htm`         | HTML parsing and text extraction                                                   |
| `EmlLoader`      | `.eml`, `.msg`          | Email files (loaded as single nodes)                                               |
| `TabularLoader`  | `.csv`, `.tsv`, `.xlsx` | Tabular data processing, preserves column structure                                |
| `CodeLoader`     | `.py`, `.js`, etc.      | Code-aware chunking with syntax preservation                                       |

#### PdfLoaderLight

A lightweight alternative to PdfLoader using pymupdf4llm for efficient PDF processing:

```python
from rakam_systems_vectorstore.components.loader import PdfLoaderLight

loader = PdfLoaderLight(
    name="pdf_loader_light",
    config={
        "chunk_size": 512,
        "chunk_overlap": 50,
        "extract_images": True,
        "image_path": "./extracted_images",
        "page_chunks": True,  # Create one chunk per page
        "write_images": True  # Save images to disk
    }
)

# Load as markdown
markdown_text = loader.load_as_text("document.pdf")

# Load as chunks (one per page or custom chunking)
chunks = loader.load_as_chunks("document.pdf")

# Load as nodes with metadata
nodes = loader.load_as_nodes("document.pdf")

# Access extracted images
image_paths = loader.get_image_paths()
for img_id, img_path in image_paths.items():
    print(f"Image {img_id}: {img_path}")
```

**Key Features:**

- Fast PDF to markdown conversion
- Optional image extraction and saving
- Page-aware chunking
- Thread-safe operations
- Lower memory footprint than PdfLoader

#### Image Extraction Support

Multiple loaders now support image extraction:

```python
from rakam_systems_vectorstore.components.loader import DocLoader, OdtLoader, PdfLoaderLight

# DocLoader with image extraction
doc_loader = DocLoader(config={
    "extract_images": True,
    "image_path": "./doc_images"
})
nodes = doc_loader.load_as_nodes("document.docx")

# Access extracted images
for img_id, img_path in doc_loader.get_image_paths().items():
    print(f"Image {img_id}: {img_path}")

# OdtLoader with image extraction
odt_loader = OdtLoader(config={
    "extract_images": True,
    "image_path": "./odt_images"
})
nodes = odt_loader.load_as_nodes("document.odt")

# PdfLoaderLight with image extraction
pdf_loader = PdfLoaderLight(config={
    "extract_images": True,
    "image_path": "./pdf_images",
    "write_images": True
})
nodes = pdf_loader.load_as_nodes("document.pdf")
```

### TextChunker

Sentence-based text chunking using Chonkie:

```python
from rakam_systems_vectorstore.components.chunker import TextChunker, create_text_chunker

chunker = TextChunker(
    chunk_size=512,        # Tokens per chunk
    chunk_overlap=50,      # Overlap in tokens
    min_sentences_per_chunk=1,
    tokenizer="character"  # Or "gpt2", HuggingFace tokenizer
)

chunks = chunker.chunk_text("Long document text...")
# Returns: [{"text": "...", "token_count": 100, "start_index": 0, "end_index": 500}, ...]

# Process multiple documents
all_chunks = chunker.run(["doc1 text", "doc2 text"])
```

### AdvancedChunker

Advanced document chunking using Docling for context-aware chunking with heading preservation:

```python
from rakam_systems_vectorstore.components.chunker import AdvancedChunker

chunker = AdvancedChunker(
    name="advanced_chunker",
    config={
        "max_tokens": 512,           # Maximum tokens per chunk
        "merge_peers": True,          # Merge peer sections
        "min_chunk_tokens": 64,       # Minimum tokens per chunk
        "filter_toc": True,           # Filter table of contents
        "include_heading_markers": True  # Include markdown headings
    }
)

# Chunk text with context preservation
chunks = chunker.chunk_text("Document text with headings...")

# Each chunk includes:
# - text: The chunk content
# - token_count: Number of tokens
# - start_index: Starting position
# - end_index: Ending position
# - heading_context: Hierarchical heading information

# Process with heading markers
chunker_with_markers = AdvancedChunker(config={
    "include_heading_markers": True
})
chunks = chunker_with_markers.chunk_text("""
# Main Title
## Section 1
Content here...
## Section 2
More content...
""")
# Output includes markdown-style headings in chunks
```

**Key Features:**

- Context-aware chunking with heading hierarchy
- Automatic merging of small chunks
- Table of contents filtering
- Image and table fragment handling
- Markdown heading markers support
- Configurable token limits and merging behavior

---

### Logging Utilities

The core package includes logging utilities:

```python
from rakam_systems_tools.utils import logging

logger = logging.getLogger(__name__)
logger.info("Processing document...")
logger.debug("Detailed debug info")
logger.error("An error occurred")
```

---

## ⚙️ Configuration System

**The Core Advantage: Configuration Without Code Changes**

Rakam Systems embraces a configuration-first approach, allowing you to modify agent behavior, vector store settings, and system parameters without touching your application code. This provides:

### Benefits of Configuration-First Design

1. **Rapid Iteration**: Test different models, prompts, or parameters instantly
2. **Environment Management**: Use different configs for dev/staging/production
3. **A/B Testing**: Compare performance of different settings by swapping configs
4. **Team Collaboration**: Non-developers can tune prompts and parameters
5. **Cost Optimization**: Switch to cheaper models for development, expensive for production
6. **Risk Reduction**: Change behavior without code deployment risks

### Real-World Scenarios

**Scenario 1: Model Optimization**

```yaml
# Week 1: Start with GPT-4o
model: openai:gpt-4o
temperature: 0.7

# Week 2: Test GPT-4o-mini for cost savings (no code changes!)
model: openai:gpt-4o-mini
temperature: 0.7

# Week 3: Back to GPT-4o for production (just revert config)
model: openai:gpt-4o
temperature: 0.5  # Also tuned temperature
```

**Scenario 2: Search Algorithm Testing**

```yaml
# Current: Using BM25 for keyword search
search:
  keyword_ranking_algorithm: bm25

# Test: Try ts_rank (just update config, no code changes!)
search:
  keyword_ranking_algorithm: ts_rank

# Decide: Compare results and keep the best one
```

**Scenario 3: Multi-Environment Deployment**

```python
# Application code (never changes)
import os
config_file = os.getenv("AGENT_CONFIG", "config/agent_config.yaml")
config = loader.load_from_yaml(config_file)
agent = loader.create_agent("my_agent", config)

# Dev: AGENT_CONFIG=config/agent_config_dev.yaml
# Staging: AGENT_CONFIG=config/agent_config_staging.yaml
# Prod: AGENT_CONFIG=config/agent_config_prod.yaml
```

## ⚙️ Configuration System Details

### VectorStoreConfig

```python
from rakam_systems_vectorstore.config import (
    VectorStoreConfig,
    EmbeddingConfig,
    DatabaseConfig,
    SearchConfig,
    IndexConfig,
    load_config
)

# Programmatic configuration
config = VectorStoreConfig(
    name="my_vectorstore",
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",
        model_name="Snowflake/snowflake-arctic-embed-m",
        batch_size=128,
        normalize=True
    ),
    database=DatabaseConfig(
        host="localhost",
        port=5432,
        database="vectorstore_db",
        user="postgres",
        password="postgres"
    ),
    search=SearchConfig(
        similarity_metric="cosine",
        default_top_k=5,
        enable_hybrid_search=True,
        hybrid_alpha=0.7
    ),
    index=IndexConfig(
        chunk_size=512,
        chunk_overlap=50,
        batch_insert_size=10000
    )
)

# From YAML file
config = VectorStoreConfig.from_yaml("config.yaml")

# From JSON file
config = VectorStoreConfig.from_json("config.json")

# From dictionary
config = VectorStoreConfig.from_dict(config_dict)

# Validation
config.validate()

# Save configuration
config.save_yaml("output_config.yaml")
config.save_json("output_config.json")
```

### YAML Configuration Example

```yaml
# vectorstore_config.yaml
name: production_vectorstore

embedding:
  model_type: sentence_transformer
  model_name: Snowflake/snowflake-arctic-embed-m
  batch_size: 128
  normalize: true

database:
  host: localhost
  port: 5432
  database: vectorstore_db
  user: postgres
  password: postgres

search:
  similarity_metric: cosine
  default_top_k: 5
  enable_hybrid_search: true
  hybrid_alpha: 0.7

index:
  chunk_size: 512
  chunk_overlap: 50
  batch_insert_size: 10000

enable_caching: true
cache_size: 1000
enable_logging: true
log_level: INFO
```

### Agent Configuration Example

```yaml
# agent_config.yaml
agents:
  my_agent:
    name: my_agent
    llm_config:
      model: openai:gpt-4o
      temperature: 0.7
      max_tokens: 2000
      parallel_tool_calls: true
    prompt_config: default_prompt
    tools:
      - search_tool
      - calculator
    deps_type: myapp.models.AgentDeps
    output_type:
      name: AgentOutput
      fields:
        answer:
          type: str
          description: The answer
        confidence:
          type: float
          description: Confidence score
    enable_tracking: true

prompts:
  default_prompt:
    system_prompt: |
      You are a helpful AI assistant.
      Always provide accurate and helpful responses.

tools:
  search_tool:
    name: search_tool
    type: direct
    module: myapp.tools
    function: search
    description: Search for information
    json_schema:
      type: object
      properties:
        query:
          type: string
      required: [query]

  calculator:
    name: calculator
    type: direct
    module: myapp.tools
    function: calculate
    description: Perform calculations
```

---

## 🚀 Quick Start Examples

### Basic Agent

```python
import asyncio
from rakam_systems_agent import BaseAgent

async def main():
    agent = BaseAgent(
        name="assistant",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant."
    )

    result = await agent.arun("What is the capital of France?")
    print(result.output_text)

asyncio.run(main())
```

### Agent with Tools

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_core.interfaces.tool import ToolComponent

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny, 25°C"

weather_tool = ToolComponent.from_function(
    function=get_weather,
    name="get_weather",
    description="Get the current weather for a city",
    json_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
)

async def main():
    agent = BaseAgent(
        name="weather_assistant",
        model="openai:gpt-4o",
        system_prompt="You help users with weather information.",
        tools=[weather_tool]
    )

    result = await agent.arun("What's the weather in Paris?")
    print(result.output_text)

asyncio.run(main())
```

### Document Search Pipeline

```python
from rakam_systems_vectorstore import (
    ConfigurablePgVectorStore,
    VectorStoreConfig,
    AdaptiveLoader
)

# Configure vector store
config = VectorStoreConfig()
store = ConfigurablePgVectorStore(config=config)
store.setup()

# Load documents
loader = AdaptiveLoader(config={"chunk_size": 512})
nodes = loader.load_as_nodes("documents/research_paper.pdf")

# Add to vector store
store.add_nodes(nodes)

# Search
results = store.search("What are the main findings?", top_k=5)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:200]}...")
    print("---")

store.shutdown()
```

### Full RAG Pipeline

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_vectorstore import ConfigurablePgVectorStore, AdaptiveLoader, VectorStoreConfig
from rakam_systems_core.interfaces.tool import ToolComponent

# Setup vector store
config = VectorStoreConfig()
store = ConfigurablePgVectorStore(config=config)
store.setup()

# Index documents
loader = AdaptiveLoader()
for doc_path in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
    nodes = loader.load_as_nodes(doc_path)
    store.add_nodes(nodes)

# Create search tool
def search_documents(query: str, top_k: int = 5) -> str:
    results = store.search(query, top_k=top_k)
    return "\n\n".join([r['content'] for r in results])

search_tool = ToolComponent.from_function(
    function=search_documents,
    name="search_documents",
    description="Search the document database",
    json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results"}
        },
        "required": ["query"]
    }
)

# Create RAG agent
async def main():
    agent = BaseAgent(
        name="rag_agent",
        model="openai:gpt-4o",
        system_prompt="""You are a helpful assistant with access to a document database.
        Use the search_documents tool to find relevant information before answering questions.""",
        tools=[search_tool]
    )

    result = await agent.arun("What are the key points from the documents?")
    print(result.output_text)

asyncio.run(main())
store.shutdown()
```

---

## Environment Variables

The system supports the following environment variables:

| Variable            | Description         | Used By                               |
| ------------------- | ------------------- | ------------------------------------- |
| `OPENAI_API_KEY`    | OpenAI API key      | OpenAIGateway, ConfigurableEmbeddings |
| `MISTRAL_API_KEY`   | Mistral API key     | MistralGateway                        |
| `COHERE_API_KEY`    | Cohere API key      | ConfigurableEmbeddings                |
| `POSTGRES_HOST`     | PostgreSQL host     | DatabaseConfig                        |
| `POSTGRES_PORT`     | PostgreSQL port     | DatabaseConfig                        |
| `POSTGRES_DB`       | PostgreSQL database | DatabaseConfig                        |
| `POSTGRES_USER`     | PostgreSQL user     | DatabaseConfig                        |
| `POSTGRES_PASSWORD` | PostgreSQL password | DatabaseConfig                        |

---

## ✅ Best Practices

1. **Always use context managers** or explicit `setup()`/`shutdown()` for proper resource management
2. **Use configuration files** for production deployments instead of hardcoded values
3. **Enable tracking** during development for debugging and evaluation
4. **Use model-specific tables** (default) to prevent mixing incompatible vector spaces
5. **Batch operations** when processing large document collections
6. **Use async methods** (`arun`, `astream`) for agents as they are powered by Pydantic AI
7. **Validate configurations** before deployment using `config.validate()` or `loader.validate_config()`

---

## 📚 Further Reading

- Example configurations: `examples/configs/`
- Agent examples: `examples/ai_agents_examples/`
- Vector store examples: `examples/ai_vectorstore_examples/`
- Loader documentation: `rakam-system-vectorstore/src/rakam_systems_vectorstore/components/loader/docs/`
- Architecture documentation: `rakam-system-vectorstore/src/rakam_systems_vectorstore/docs/ARCHITECTURE.md`

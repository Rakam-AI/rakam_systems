# Rakam Systems Components Guide

Welcome! This guide introduces the modular components of Rakam Systems, a framework for building robust, production-ready AI applications. Here, you'll find clear explanations and practical examples for using agents, vector stores, and configuration systems.

---

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

Rakam Systems is split into three main packages:

- **Core**: Abstractions, interfaces, and base classes
- **Agent**: AI agent implementations (depends on core)
- **Vectorstore**: Vector storage and document processing (depends on core)

**Design Highlights:**

- Modular: Install only what you need
- Clear dependencies: Agent/vectorstore depend on core
- Component-based: All components extend `BaseComponent` with lifecycle methods (`setup()`, `shutdown()`)
- Interface-driven: Abstract interfaces for easy extension
- Configuration-first: YAML/JSON config for all components
- Provider-agnostic: Works with many LLMs, embedding models, and vector stores

---

---

## Core Package (`rakam-systems-core`)

The core package provides the foundational building blocks for Rakam Systems. Install this first before using agents or vector stores.

**Key Features:**

- Base classes for all components
- Abstract interfaces for agents, tools, LLM gateways, vector stores, and loaders
- Built-in tracking for debugging and evaluation
- Configuration loader for YAML/JSON configs

**Example: Using the BaseComponent**

```python
from rakam_system_core.base import BaseComponent

class MyComponent(BaseComponent):
    def setup(self):
        # Initialization logic
        pass
    def shutdown(self):
        # Cleanup logic
        pass
```

**Tracking Example:**

```python
from rakam_systems_core.tracking import TrackingMixin, track_method

class MyAgent(TrackingMixin, BaseAgent):
    @track_method()
    def run(self, input):
        ...

# Enable tracking
agent.enable_tracking(output_dir="./tracking")
agent.export_tracking_data(format='csv')
```

**Configuration Loader Example:**

```python
from rakam_systems_core.config_loader import ConfigurationLoader

loader = ConfigurationLoader()
config = loader.load_from_yaml("agent_config.yaml")
agent = loader.create_agent("my_agent", config)
```

---

## 🤖 Agent Package (`rakam-system-agent`)

The agent package provides ready-to-use AI agent implementations powered by Pydantic AI. Install with:

```bash
pip install rakam-systems-agent
```

(Requires the core package.)

**Key Features:**

- BaseAgent class for building and running agents
- Dynamic system prompts for context-aware responses
- Multiple chat history backends (JSON, SQLite, PostgreSQL)
- LLM gateway support (OpenAI, Mistral, and more)

**Example: Running an Agent**

```python
from rakam_systems_agent import BaseAgent

agent = BaseAgent(name="my_agent", enable_tracking=True)
result = await agent.arun("What is AI?")
print(result.output_text)
```

**Dynamic System Prompts:**

Inject context at runtime using decorators or direct registration:

```python
@agent.dynamic_system_prompt

    return f"Today's date is {date.today().strftime('%B %d, %Y')}."

agent.add_dynamic_system_prompt(add_time_context)
```

---

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

The vectorstore package provides vector database and document processing tools. Install with:

```bash
pip install -e ./rakam-system-vectorstore
```

(Requires the core package.)

**Key Features:**

- Store and search document embeddings
- Hybrid, vector, and keyword search
- Multi-backend embedding support (OpenAI, Cohere, HuggingFace, etc.)
- Adaptive file loaders for many formats

**Example: Using the Vector Store**

```python
from rakam_systems_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

config = VectorStoreConfig()
store = ConfigurablePgVectorStore(config=config)
store.setup()
results = store.search("What is machine learning?", top_k=5)
```

---

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

Rakam Systems is configuration-first: you can change agent behavior, vector store settings, and more—without touching your code.

**Why use configuration?**

- Rapidly test different models, prompts, or parameters
- Manage dev/staging/production environments easily
- Enable A/B testing and team collaboration
- Optimize costs and reduce deployment risk

**Example: Switching Models with YAML**

```yaml
# Week 1: Use GPT-4o
model: openai:gpt-4o
temperature: 0.7

# Week 2: Try GPT-4o-mini (no code changes!)
model: openai:gpt-4o-mini
temperature: 0.7
```

**Example: Programmatic Configuration**

```python
from rakam_systems_vectorstore.config import VectorStoreConfig

config = VectorStoreConfig(name="my_vectorstore")
config.save_yaml("output_config.yaml")
```

---

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

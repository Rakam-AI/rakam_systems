# AI VectorStore

A modular, production-ready vector store system for semantic search and retrieval-augmented generation (RAG) applications. Part of the Rakam Systems AI framework.

## Overview

The `ai_vectorstore` package provides a comprehensive, production-ready set of components for building vector-based search and retrieval systems. It supports multiple backend implementations (PostgreSQL with pgvector, FAISS) and includes all necessary components for a complete RAG pipeline.

### What's New

- ✨ **ConfigurablePgVectorStore**: Enhanced vector store with full YAML/JSON configuration support
- ⚙️ **Configuration System**: Centralized configuration management with validation and environment variable support
- 🔄 **Update Operations**: Update existing vectors, embeddings, and metadata without re-indexing
- 🔌 **Pluggable Embeddings**: Support for multiple embedding providers (SentenceTransformers, OpenAI, Cohere)
- 📊 **Enhanced Search**: Multiple similarity metrics (cosine, L2, dot product) with configurable hybrid search
- 🎯 **Better DX**: Improved developer experience with clearer APIs and comprehensive documentation

### Quick Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configurable Vector Store (Recommended)](#configurable-vector-store-recommended)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Migration Guide](#migration-guide)

## Features

- **Multiple Vector Store Backends**
  - PostgreSQL with pgvector (persistent, production-ready)
  - FAISS (in-memory, high-performance)
  - Configurable vector store with YAML/JSON configuration support
  
- **Hybrid Search Capabilities**
  - Vector similarity search (cosine, L2, dot product)
  - Full-text search (PostgreSQL)
  - Combined hybrid search with configurable weighting
  - Adjustable alpha parameter for search balance
  
- **Advanced Retrieval**
  - Built-in re-ranking for improved relevance
  - Metadata filtering with Django ORM support
  - Collection-based organization
  - LRU caching for query performance optimization
  - Update operations for existing vectors
  
- **Flexible Embedding Options**
  - Local embedding models (SentenceTransformers)
  - OpenAI API integration
  - Cohere API support
  - Configurable embedding dimensions
  - Pluggable embedding backends

- **Complete RAG Pipeline Components**
  - Document loaders (file, adaptive)
  - Text chunkers (simple, text-based)
  - Embedding models (configurable, OpenAI)
  - Indexers (simple indexer)
  - Retrievers (basic retriever)
  - Re-rankers (model-based)

- **Configuration Management**
  - YAML/JSON configuration files
  - Environment variable overrides
  - Programmatic configuration
  - Configuration validation and defaults

## Architecture

The package follows a modular architecture with clear interfaces:

```
ai_vectorstore/
├── core.py                           # Core data structures (VSFile, Node, NodeMetadata)
├── config.py                         # Configuration management system
├── components/
│   ├── vectorstore/                  # Vector store implementations
│   │   ├── pg_vector_store.py            # PostgreSQL backend
│   │   ├── configurable_pg_vector_store.py  # Enhanced configurable backend
│   │   ├── faiss_vector_store.py         # FAISS backend
│   │   ├── pg_models.py                  # Django ORM models
│   │   └── migrations/                   # Database migrations
│   ├── chunker/                      # Text chunking components
│   │   ├── simple_chunker.py             # Basic text chunker
│   │   └── text_chunker.py               # Advanced text chunker
│   ├── embedding_model/              # Embedding generation
│   │   ├── configurable_embeddings.py    # Pluggable embedding system
│   │   └── openai_embeddings.py          # OpenAI API embeddings
│   ├── indexer/                      # Document indexing
│   │   └── simple_indexer.py             # Basic indexer
│   ├── loader/                       # Document loading
│   │   ├── file_loader.py                # File loading
│   │   └── adaptive_loader.py            # Adaptive loading
│   ├── reranker/                     # Result re-ranking
│   │   └── model_reranker.py             # Model-based reranker
│   └── retriever/                    # Search and retrieval
│       └── basic_retriever.py            # Basic retriever
└── server/                           # MCP server implementation
    └── mcp_server_vector.py
```

## Installation

### As Part of Rakam Systems (Recommended)

```bash
cd app/rakam_systems

# Full AI Vectorstore with all features
pip install -e ".[ai-vectorstore]"
```

### Standalone Installation

For granular control over dependencies:

```bash
cd app/rakam_systems/rakam_systems/ai_vectorstore

# Core only (minimal)
pip install -e .

# With PostgreSQL backend
pip install -e ".[postgres]"

# With FAISS backend
pip install -e ".[faiss]"

# With local embeddings (SentenceTransformers)
pip install -e ".[local-embeddings]"

# With OpenAI or Cohere embeddings
pip install -e ".[openai]"  # or ".[cohere]"

# With document loaders
pip install -e ".[loaders]"

# Everything
pip install -e ".[all]"
```

**📖 See [INSTALLATION.md](INSTALLATION.md) for complete installation guide** | **⚡ [QUICK_INSTALL.md](QUICK_INSTALL.md) for quick reference**

## Quick Start

### PostgreSQL Vector Store

```python
from ai_vectorstore.components.vectorstore.pg_vector_store import PgVectorStore
from ai_vectorstore.core import Node, NodeMetadata, VSFile

# Initialize the vector store
vector_store = PgVectorStore(
    name="my_vector_store",
    embedding_model="Snowflake/snowflake-arctic-embed-m",
    use_embedding_api=False  # Use local model
)

# Create a collection
collection_name = "documents"
vector_store.create_collection(collection_name)

# Add documents
nodes = [
    Node(
        content="Your document content here",
        metadata=NodeMetadata(
            source_file_uuid="file-uuid",
            position=0,
            custom={"title": "Document Title"}
        )
    )
]
vector_store.add_nodes(nodes, collection_name)

# Search
results = vector_store.search(
    query="Your search query",
    collection_name=collection_name,
    top_k=5,
    search_type="hybrid"  # or "vector", "fts"
)
```

### FAISS Vector Store

```python
from ai_vectorstore.components.vectorstore.faiss_vector_store import FaissStore
from ai_vectorstore.core import Node, NodeMetadata

# Initialize FAISS store
faiss_store = FaissStore(
    name="my_faiss_store",
    base_index_path="./faiss_indexes",
    embedding_model="Snowflake/snowflake-arctic-embed-m"
)

# Create collection and add nodes
collection_name = "documents"
faiss_store.create_collection(collection_name)

nodes = [
    Node(
        content="Your content here",
        metadata=NodeMetadata(
            source_file_uuid="file-uuid",
            position=0
        )
    )
]
faiss_store.add_nodes(nodes, collection_name)

# Search
results = faiss_store.query_nodes(
    query="search query",
    collection_name=collection_name,
    top_k=5
)
```

## Configurable Vector Store (Recommended)

The `ConfigurablePgVectorStore` is an enhanced, production-ready vector store that supports configuration via YAML/JSON files or dictionaries. This is the recommended approach for production deployments.

### Configuration System

The configuration system provides a unified way to manage all vector store settings:

```python
from ai_vectorstore.config import VectorStoreConfig, EmbeddingConfig, SearchConfig
from ai_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore

# Option 1: Use defaults
vector_store = ConfigurablePgVectorStore()

# Option 2: Load from YAML file
vector_store = ConfigurablePgVectorStore(
    name="my_store",
    config="/path/to/config.yaml"
)

# Option 3: Programmatic configuration
config = VectorStoreConfig(
    name="custom_store",
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",
        model_name="Snowflake/snowflake-arctic-embed-m",
        batch_size=32
    ),
    search=SearchConfig(
        similarity_metric="cosine",
        default_top_k=5,
        enable_hybrid_search=True,
        hybrid_alpha=0.7,
        rerank=True
    )
)
vector_store = ConfigurablePgVectorStore(name="my_store", config=config)
```

### YAML Configuration Example

Create a `vectorstore_config.yaml` file:

```yaml
name: production_vectorstore

embedding:
  model_type: sentence_transformer
  model_name: Snowflake/snowflake-arctic-embed-m
  batch_size: 32
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
  rerank: true
  search_buffer_factor: 2

index:
  chunk_size: 512
  chunk_overlap: 50
  batch_insert_size: 100

enable_caching: true
cache_size: 1000
enable_logging: true
log_level: INFO
```

### Using ConfigurablePgVectorStore

```python
from ai_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore
from ai_vectorstore.core import Node, NodeMetadata

# Initialize with config
store = ConfigurablePgVectorStore(
    name="my_vectorstore",
    config="config.yaml"
)
store.setup()

# Create collection
store.get_or_create_collection("documents")

# Add nodes
nodes = [
    Node(
        content="Document content here",
        metadata=NodeMetadata(
            source_file_uuid="file-123",
            position=0,
            custom={"title": "Document Title"}
        )
    )
]
store.add_nodes("documents", nodes)

# Search with configuration defaults
results, nodes = store.search(
    collection_name="documents",
    query="search query"
)

# Override configuration at search time
results, nodes = store.search(
    collection_name="documents",
    query="search query",
    distance_type="l2",
    number=10,
    hybrid_search=False
)

# Update existing vectors
store.update_vector(
    collection_name="documents",
    node_id=1,
    new_content="Updated content",  # Will regenerate embedding
    new_metadata={"status": "reviewed"}
)

# Delete specific nodes
store.delete_nodes("documents", [1, 2, 3])

# Get collection information
info = store.get_collection_info("documents")
print(f"Collection has {info['node_count']} nodes")
```

### Configuration Classes

**EmbeddingConfig** - Embedding model configuration:
- `model_type`: "sentence_transformer", "openai", "cohere"
- `model_name`: Model identifier
- `api_key`: API key (auto-loaded from environment)
- `batch_size`: Batch size for embeddings
- `normalize`: Normalize embeddings
- `dimensions`: Embedding dimensions (auto-detected)

**DatabaseConfig** - Database connection settings:
- `host`, `port`, `database`, `user`, `password`
- `pool_size`, `max_overflow`: Connection pooling
- Auto-loads from `POSTGRES_*` environment variables

**SearchConfig** - Search behavior configuration:
- `similarity_metric`: "cosine", "l2", "dot_product"
- `default_top_k`: Default number of results
- `enable_hybrid_search`: Enable hybrid search
- `hybrid_alpha`: Vector/keyword balance (0-1)
- `rerank`: Enable re-ranking
- `search_buffer_factor`: Buffer for re-ranking

**IndexConfig** - Indexing configuration:
- `chunk_size`, `chunk_overlap`: Chunking parameters
- `enable_parallel_processing`: Parallel processing
- `parallel_workers`: Number of workers
- `batch_insert_size`: Batch size for inserts

## Core Data Structures

### VSFile
Represents a source file to be processed:
```python
from ai_vectorstore.core import VSFile

file = VSFile(file_path="/path/to/document.pdf")
# Automatically extracts filename, MIME type, and UUID
```

### Node
Represents a chunk of content with metadata:
```python
from ai_vectorstore.core import Node, NodeMetadata

node = Node(
    content="Document chunk text",
    metadata=NodeMetadata(
        source_file_uuid="file-uuid",
        position=0,  # page number or chunk position
        custom={"author": "John Doe", "date": "2025-01-01"}
    )
)
```

### NodeMetadata
Stores metadata about each node:
- `node_id`: Unique identifier (auto-assigned)
- `source_file_uuid`: Reference to source file
- `position`: Position in source (page number, chunk index, etc.)
- `custom`: Dictionary for arbitrary metadata

## PostgreSQL Backend

### Features
- ✅ Persistent storage with ACID transactions
- ✅ Hybrid search (vector + full-text)
- ✅ Built-in re-ranking with cross-encoder models
- ✅ Metadata filtering and custom queries
- ✅ Collection-based organization
- ✅ Query result caching (LRU)
- ✅ Django ORM integration
- ✅ Update operations (content, embeddings, metadata)
- ✅ Multiple similarity metrics (cosine, L2, dot product)
- ✅ Configuration-driven architecture

### Setup

1. **Start PostgreSQL with pgvector:**
```bash
docker run -d \
  --name postgres-vectorstore \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vectorstore_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

2. **Configure Django settings:**
```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_app.settings')

# In your Django settings:
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'vectorstore_db',
        'USER': 'postgres',
        'PASSWORD': 'postgres',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

INSTALLED_APPS = [
    'ai_vectorstore.components.vectorstore',
    # ... other apps
]
```

3. **Run migrations:**
```bash
python manage.py migrate
```

### Search Types

**Vector Search** - Pure semantic similarity:
```python
results = vector_store.search(
    query="machine learning",
    collection_name="docs",
    search_type="vector",
    top_k=5
)
```

**Full-Text Search** - Traditional keyword matching:
```python
results = vector_store.search(
    query="exact phrase match",
    collection_name="docs",
    search_type="fts",
    top_k=5
)
```

**Hybrid Search** - Combined approach (best results):
```python
results = vector_store.search(
    query="machine learning algorithms",
    collection_name="docs",
    search_type="hybrid",
    alpha=0.5,  # 0=pure FTS, 1=pure vector
    top_k=5,
    rerank=True  # Enable cross-encoder re-ranking
)
```

### Advanced Features

**Metadata Filtering:**
```python
# Custom filtering with Django ORM
from ai_vectorstore.components.vectorstore.pg_models import NodeEntry

filtered_results = vector_store.search(
    query="query text",
    collection_name="docs",
    filter_func=lambda qs: qs.filter(
        metadata__custom__author="John Doe"
    )
)
```

**Batch Operations:**
```python
# Batch add nodes
vector_store.add_nodes_batch(
    nodes_list=all_nodes,
    collection_name="docs",
    batch_size=100
)
```

**Collection Management:**
```python
# List collections
collections = vector_store.list_collections()

# Delete collection
vector_store.delete_collection("old_collection")

# Get collection stats (PgVectorStore)
stats = vector_store.get_collection_stats("docs")

# Get collection info (ConfigurablePgVectorStore)
info = vector_store.get_collection_info("docs")
print(f"Nodes: {info['node_count']}, Dimensions: {info['embedding_dim']}")
```

**Update Operations (ConfigurablePgVectorStore):**
```python
from ai_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore

store = ConfigurablePgVectorStore(config="config.yaml")

# Update content (will regenerate embedding automatically)
store.update_vector(
    collection_name="docs",
    node_id=123,
    new_content="Updated document content",
    new_metadata={"status": "reviewed", "version": 2}
)

# Update only embedding (e.g., with improved model)
new_embedding = embedding_model.encode("document content")
store.update_vector(
    collection_name="docs",
    node_id=123,
    new_embedding=new_embedding
)

# Update only metadata
store.update_vector(
    collection_name="docs",
    node_id=123,
    new_metadata={"priority": "high"}
)

# Delete specific nodes
store.delete_nodes(collection_name="docs", node_ids=[123, 456, 789])
```

## FAISS Backend

### Features
- ✅ In-memory, extremely fast search
- ✅ No database required
- ✅ Persistent index saving/loading
- ✅ Collection-based organization
- ✅ Perfect for prototyping and development

### Usage

```python
from ai_vectorstore.components.vectorstore.faiss_vector_store import FaissStore

# Initialize
store = FaissStore(
    base_index_path="./indexes",
    embedding_model="Snowflake/snowflake-arctic-embed-m"
)

# Operations are similar to PostgreSQL backend
store.create_collection("docs")
store.add_nodes(nodes, "docs")
results = store.query_nodes("query", "docs", top_k=5)

# Save/load indexes
store.save_vector_store()  # Auto-saves to base_index_path
store.load_vector_store()  # Loads on init if not initialising=True
```

## Configuration

### Environment Variables

The configuration system automatically loads values from environment variables:

```bash
# Embedding API Keys
export OPENAI_API_KEY="your-openai-api-key"
export COHERE_API_KEY="your-cohere-api-key"

# PostgreSQL Connection (auto-loaded by DatabaseConfig)
export POSTGRES_DB="vectorstore_db"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"

# Django Settings
export DJANGO_SETTINGS_MODULE="your_app.settings"
```

### Loading and Saving Configurations

```python
from ai_vectorstore.config import VectorStoreConfig, load_config

# Load from file (auto-detects format)
config = load_config("config.yaml")  # or config.json

# Load from dictionary
config_dict = {
    "name": "my_store",
    "embedding": {"model_name": "custom-model"},
    "search": {"default_top_k": 10}
}
config = load_config(config_dict)

# Validate configuration
config.validate()

# Save to file
config.save_yaml("output_config.yaml")
config.save_json("output_config.json")

# Convert to dictionary
config_dict = config.to_dict()
```

### Embedding Models

**Local Models (SentenceTransformers):**
- `Snowflake/snowflake-arctic-embed-m` (default, 768-dim, recommended)
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast)
- `BAAI/bge-large-en-v1.5` (1024-dim, high quality)
- `sentence-transformers/all-mpnet-base-v2` (768-dim, balanced)

**OpenAI API:**
- `text-embedding-3-small` (1536-dim)
- `text-embedding-3-large` (3072-dim)
- `text-embedding-ada-002` (1536-dim, legacy)

**Cohere API:**
- `embed-english-v3.0`
- `embed-multilingual-v3.0`

**Legacy PgVectorStore (deprecated):**
```python
# Use local model
vector_store = PgVectorStore(
    embedding_model="Snowflake/snowflake-arctic-embed-m",
    use_embedding_api=False
)

# Use OpenAI API
vector_store = PgVectorStore(
    use_embedding_api=True,
    api_model="text-embedding-3-small"
)
```

**ConfigurablePgVectorStore (recommended):**
```python
from ai_vectorstore.config import VectorStoreConfig, EmbeddingConfig
from ai_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore

# Local SentenceTransformer model
config = VectorStoreConfig(
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",
        model_name="Snowflake/snowflake-arctic-embed-m",
        batch_size=32,
        normalize=True
    )
)
vector_store = ConfigurablePgVectorStore(config=config)

# OpenAI API
config = VectorStoreConfig(
    embedding=EmbeddingConfig(
        model_type="openai",
        model_name="text-embedding-3-small",
        api_key="your-api-key"  # Or set OPENAI_API_KEY env var
    )
)
vector_store = ConfigurablePgVectorStore(config=config)

# Cohere API
config = VectorStoreConfig(
    embedding=EmbeddingConfig(
        model_type="cohere",
        model_name="embed-english-v3.0",
        api_key="your-api-key"  # Or set COHERE_API_KEY env var
    )
)
vector_store = ConfigurablePgVectorStore(config=config)
```

## Examples

Comprehensive examples are available in the `examples/ai_vectorstore_examples/` directory:

- **`postgres_vectorstore_example.py`** - Full PostgreSQL implementation with hybrid search
- **`basic_faiss_example.py`** - FAISS in-memory vector store
- **`run_postgres_example.sh`** - One-command setup script

See [examples/ai_vectorstore_examples/README.md](../../examples/ai_vectorstore_examples/README.md) for detailed documentation.

## Performance Considerations

### PostgreSQL
- **Indexing**: Uses pgvector's HNSW (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search
- **Caching**: LRU cache for query results (configurable cache size)
- **Batch Operations**: Use `add_nodes_batch()` for bulk inserts
- **Connection Pooling**: Configure Django database connection pooling for production

### FAISS
- **Index Types**: Automatically uses IndexFlatL2 for accuracy; can be customized for larger datasets
- **Memory Usage**: Entire index in memory; monitor RAM usage with large datasets
- **Persistence**: Save/load operations can be expensive; use sparingly

## Best Practices

1. **Choose the Right Backend:**
   - Use `ConfigurablePgVectorStore` for production deployments (recommended)
   - Use legacy `PgVectorStore` for backward compatibility
   - Use FAISS for prototyping, maximum speed, and memory-based applications
   - Leverage configuration files for reproducible deployments

2. **Configuration Management:**
   - Use YAML/JSON configuration files for production
   - Store configurations in version control
   - Use environment variables for secrets (API keys, passwords)
   - Validate configurations before deployment
   - Document configuration changes

3. **Optimize Embedding Models:**
   - Start with `Snowflake/snowflake-arctic-embed-m` for balanced performance
   - Use smaller models (384-dim) for speed-critical applications
   - Use larger models (1024-dim+) for maximum accuracy
   - Consider API-based models (OpenAI, Cohere) for convenience
   - Batch encode documents for better throughput

4. **Leverage Hybrid Search:**
   - Use `hybrid_alpha=0.7` as starting point (favors vector search)
   - Tune alpha based on your specific use case:
     - 0.9-1.0: Pure semantic search
     - 0.5-0.7: Balanced approach
     - 0.0-0.3: Keyword-focused search
   - Enable re-ranking for best results (adds latency but improves relevance)
   - Set `search_buffer_factor=2` to retrieve more candidates for re-ranking

5. **Metadata Design:**
   - Store searchable metadata in custom fields
   - Use consistent metadata structure across documents
   - Index frequently queried fields
   - Include version and timestamp metadata
   - Use metadata for filtering and access control

6. **Chunking Strategy:**
   - Keep chunks between 200-500 tokens for optimal retrieval
   - Include overlap (50-100 tokens) between chunks for context preservation
   - Store chunk position for result ordering
   - Consider semantic chunking for better coherence
   - Configure `chunk_size` and `chunk_overlap` in IndexConfig

7. **Update and Maintenance:**
   - Use `update_vector()` for content corrections without re-indexing
   - Periodically update embeddings with improved models
   - Monitor and clean up outdated nodes
   - Use batch operations for large-scale updates
   - Keep track of node IDs for efficient updates

8. **Performance Optimization:**
   - Enable caching with appropriate `cache_size` (default: 1000)
   - Use batch inserts with optimal `batch_insert_size` (default: 100)
   - Monitor database connection pool settings
   - Consider parallel processing for large datasets
   - Use appropriate similarity metric for your use case:
     - `cosine`: Normalized vectors (recommended for most cases)
     - `l2`: Euclidean distance (good for non-normalized vectors)
     - `dot_product`: Fast but requires normalized vectors

## Integration with RAG Pipeline

```python
from ai_vectorstore.components.loader.file_loader import FileLoader
from ai_vectorstore.components.chunker.simple_chunker import SimpleChunker
from ai_vectorstore.components.embedding_model.openai_embeddings import OpenAIEmbeddings
from ai_vectorstore.components.vectorstore.pg_vector_store import PgVectorStore
from ai_vectorstore.components.retriever.basic_retriever import BasicRetriever

# 1. Load documents
loader = FileLoader()
documents = loader.load("/path/to/documents")

# 2. Chunk documents
chunker = SimpleChunker()
chunks = chunker.run(documents)

# 3. Generate embeddings and store
vector_store = PgVectorStore()
vector_store.create_collection("knowledge_base")
vector_store.add_nodes(chunks, "knowledge_base")

# 4. Retrieve relevant context
retriever = BasicRetriever(vector_store=vector_store)
relevant_docs = retriever.retrieve(
    query="user question",
    collection_name="knowledge_base",
    top_k=5
)

# 5. Use with your LLM
# context = "\n\n".join([doc.content for doc in relevant_docs])
# response = llm.generate(prompt + context)
```

## Troubleshooting

### PostgreSQL Connection Issues
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker logs postgres-vectorstore

# Verify connection
psql -h localhost -U postgres -d vectorstore_db
```

### Embedding Model Issues
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub

# Verify GPU availability (for GPU models)
python -c "import torch; print(torch.cuda.is_available())"
```

### Migration Issues
```bash
# Reset migrations (development only!)
python manage.py migrate ai_vectorstore zero
python manage.py migrate
```

## API Reference

### ConfigurablePgVectorStore (Recommended)

**Initialization:**
- `__init__(name: str, config: Union[VectorStoreConfig, Dict, str])` - Initialize with configuration
- `setup()` - Setup resources and connections

**Collection Management:**
- `get_or_create_collection(collection_name: str, embedding_dim: int)` - Get or create collection
- `create_collection_from_nodes(collection_name: str, nodes: List[Node])` - Create from nodes
- `create_collection_from_files(collection_name: str, files: List[VSFile])` - Create from files
- `list_collections()` - List all collections
- `get_collection_info(collection_name: str)` - Get collection information
- `delete_collection(collection_name: str)` - Delete a collection

**Node Operations:**
- `add_nodes(collection_name: str, nodes: List[Node])` - Add nodes to collection
- `update_vector(collection_name: str, node_id: int, new_content: str, new_embedding: List[float], new_metadata: Dict)` - Update existing vector
- `delete_nodes(collection_name: str, node_ids: List[int])` - Delete specific nodes

**Search:**
- `search(collection_name: str, query: str, distance_type: str, number: int, meta_data_filters: Dict, hybrid_search: bool)` - Search collection with hybrid support
- `query(vector: List[float], top_k: int, **kwargs)` - Query by vector (VectorStore interface)

**Lifecycle:**
- `shutdown()` - Cleanup and shutdown

### PgVectorStore (Legacy)
- `create_collection(name: str)` - Create a new collection
- `add_nodes(nodes: List[Node], collection_name: str)` - Add nodes to collection
- `search(query: str, collection_name: str, search_type: str, top_k: int)` - Search collection
- `delete_collection(name: str)` - Delete a collection
- `list_collections()` - List all collections
- `get_collection_stats(name: str)` - Get collection statistics

### FaissStore
- `create_collection(name: str)` - Create a new collection
- `add_nodes(nodes: List[Node], collection_name: str)` - Add nodes to collection
- `query_nodes(query: str, collection_name: str, top_k: int)` - Query collection
- `save_vector_store()` - Save indexes to disk
- `load_vector_store()` - Load indexes from disk

### Configuration API

**VectorStoreConfig:**
- `from_dict(config_dict: Dict)` - Create from dictionary
- `from_yaml(yaml_path: str)` - Load from YAML file
- `from_json(json_path: str)` - Load from JSON file
- `to_dict()` - Convert to dictionary
- `save_yaml(output_path: str)` - Save to YAML file
- `save_json(output_path: str)` - Save to JSON file
- `validate()` - Validate configuration

**load_config:**
- `load_config(config_source: Union[str, Dict], config_type: str)` - Universal config loader

## Migration Guide

### From PgVectorStore to ConfigurablePgVectorStore

If you're using the legacy `PgVectorStore`, here's how to migrate to the new `ConfigurablePgVectorStore`:

**Before (Legacy):**
```python
from ai_vectorstore.components.vectorstore.pg_vector_store import PgVectorStore

vector_store = PgVectorStore(
    name="my_store",
    embedding_model="Snowflake/snowflake-arctic-embed-m",
    use_embedding_api=False
)

vector_store.create_collection("docs")
vector_store.add_nodes(nodes, "docs")
results = vector_store.search(
    query="test",
    collection_name="docs",
    search_type="hybrid",
    top_k=5
)
```

**After (Configurable):**
```python
from ai_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore
from ai_vectorstore.config import VectorStoreConfig, EmbeddingConfig

# Option 1: Simple migration with defaults
vector_store = ConfigurablePgVectorStore(name="my_store")
vector_store.setup()

# Option 2: With explicit configuration
config = VectorStoreConfig(
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",
        model_name="Snowflake/snowflake-arctic-embed-m"
    )
)
vector_store = ConfigurablePgVectorStore(name="my_store", config=config)
vector_store.setup()

# Collections are auto-created
vector_store.get_or_create_collection("docs")
vector_store.add_nodes("docs", nodes)

# Search (note: different return format)
results_dict, result_nodes = vector_store.search(
    collection_name="docs",
    query="test",
    hybrid_search=True,
    number=5
)
```

**Key Differences:**
1. Must call `setup()` after initialization
2. Use `get_or_create_collection()` instead of `create_collection()`
3. `search()` returns tuple of (results_dict, nodes_list)
4. Search parameters: `top_k` → `number`, `search_type` → `hybrid_search` (bool)
5. Configuration is centralized and supports YAML/JSON files
6. New features: `update_vector()`, `delete_nodes()`, `get_collection_info()`

## Contributing

This package is part of the Rakam Systems framework. For contributions and development:

1. Follow the existing interface patterns in `ai_core.interfaces`
2. Add comprehensive docstrings and type hints
3. Include examples in the `examples/` directory
4. Update this README with new features
5. Test with both legacy and configurable vector stores
6. Ensure backward compatibility when possible

## License

See [LICENSE](LICENSE) file for details.

## Additional Resources

### Rakam Systems Documentation
- [Main Rakam Systems Documentation](../../README.md)
- [AI Core Interfaces](../ai_core/interfaces/)
- [AI Agents Framework](../ai_agents/)
- [Examples Directory](../../examples/ai_vectorstore_examples/)

### External Documentation
- [pgvector Documentation](https://github.com/pgvector/pgvector) - PostgreSQL vector extension
- [FAISS Documentation](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [SentenceTransformers Documentation](https://www.sbert.net/) - Sentence embedding models
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) - OpenAI embedding models
- [Cohere Embeddings API](https://docs.cohere.com/docs/embeddings) - Cohere embedding models

### Configuration Examples
- See [examples/configs/pg_vectorstore_config.yaml](../../examples/configs/pg_vectorstore_config.yaml) for example YAML configuration
- See [examples/ai_vectorstore_examples/](../../examples/ai_vectorstore_examples/) for working examples

### Key Files
- `config.py` - Configuration system implementation
- `core.py` - Core data structures (VSFile, Node, NodeMetadata)
- `components/vectorstore/configurable_pg_vector_store.py` - Enhanced configurable vector store
- `components/vectorstore/pg_vector_store.py` - Legacy vector store
- `components/embedding_model/configurable_embeddings.py` - Pluggable embedding system

## Support and Contributing

For issues, questions, or contributions, please refer to the main Rakam Systems repository. When reporting issues:

1. Include your configuration (sanitized of secrets)
2. Provide code snippets demonstrating the issue
3. Include error messages and stack traces
4. Specify versions of key dependencies (Django, pgvector, etc.)

## Changelog

### v1.1.0 (Latest)
- Added `ConfigurablePgVectorStore` with full configuration support
- Introduced centralized configuration system (`config.py`)
- Added `ConfigurableEmbeddings` for pluggable embedding models
- Implemented `update_vector()` for in-place updates
- Added support for multiple similarity metrics
- Enhanced hybrid search with configurable alpha parameter
- Improved documentation with migration guide

### v1.0.0
- Initial release with `PgVectorStore` and `FaissStore`
- Basic hybrid search support
- Django ORM integration
- Collection-based organization

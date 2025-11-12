# AI VectorStore

A modular, production-ready vector store system for semantic search and retrieval-augmented generation (RAG) applications. Part of the Rakam Systems AI framework.

## Overview

The `ai_vectorstore` package provides a comprehensive set of components for building vector-based search and retrieval systems. It supports multiple backend implementations (PostgreSQL with pgvector, FAISS) and includes all necessary components for a complete RAG pipeline.

## Features

- **Multiple Vector Store Backends**
  - PostgreSQL with pgvector (persistent, production-ready)
  - FAISS (in-memory, high-performance)
  
- **Hybrid Search Capabilities**
  - Vector similarity search
  - Full-text search (PostgreSQL)
  - Combined hybrid search with configurable weighting
  
- **Advanced Retrieval**
  - Built-in re-ranking for improved relevance
  - Metadata filtering
  - Collection-based organization
  - Caching for performance optimization
  
- **Flexible Embedding Options**
  - Local embedding models (SentenceTransformers)
  - OpenAI API integration
  - Customizable embedding dimensions

- **Complete RAG Pipeline Components**
  - Document loaders
  - Text chunkers
  - Embedding models
  - Indexers
  - Retrievers
  - Re-rankers

## Architecture

The package follows a modular architecture with clear interfaces:

```
ai_vectorstore/
├── core.py                    # Core data structures (VSFile, Node, NodeMetadata)
├── components/
│   ├── vectorstore/          # Vector store implementations
│   │   ├── pg_vector_store.py    # PostgreSQL backend
│   │   ├── faiss_vector_store.py # FAISS backend
│   │   └── pg_models.py          # Django ORM models
│   ├── chunker/              # Text chunking components
│   ├── embedding_model/      # Embedding generation
│   ├── indexer/              # Document indexing
│   ├── loader/               # Document loading
│   ├── reranker/             # Result re-ranking
│   └── retriever/            # Search and retrieval
└── server/                   # MCP server implementation
```

## Installation

```bash
# Install the package
pip install -e .

# For PostgreSQL support with Django
pip install django psycopg2-binary pgvector

# For FAISS support
pip install faiss-cpu  # or faiss-gpu for GPU support

# For embedding models
pip install sentence-transformers
```

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

# Get collection stats
stats = vector_store.get_collection_stats("docs")
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

```bash
# OpenAI API (if using API embeddings)
export OPENAI_API_KEY="your-api-key"

# PostgreSQL (if using PgVectorStore)
export POSTGRES_DB="vectorstore_db"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export DJANGO_SETTINGS_MODULE="your_app.settings"
```

### Embedding Models

**Local Models (SentenceTransformers):**
- `Snowflake/snowflake-arctic-embed-m` (default, 768-dim)
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast)
- `BAAI/bge-large-en-v1.5` (1024-dim, high quality)

**OpenAI API:**
- `text-embedding-3-small` (1536-dim)
- `text-embedding-3-large` (3072-dim)

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
   - Use PostgreSQL for production, persistence, and hybrid search
   - Use FAISS for prototyping, maximum speed, and memory-based applications

2. **Optimize Embedding Models:**
   - Start with `Snowflake/snowflake-arctic-embed-m` for balanced performance
   - Use smaller models (384-dim) for speed-critical applications
   - Use larger models (1024-dim+) for maximum accuracy

3. **Leverage Hybrid Search:**
   - Use `alpha=0.5` as starting point for hybrid search
   - Tune alpha based on your specific use case
   - Enable re-ranking for best results (adds latency)

4. **Metadata Design:**
   - Store searchable metadata in custom fields
   - Use consistent metadata structure across documents
   - Index frequently queried fields

5. **Chunking Strategy:**
   - Keep chunks between 200-500 tokens for optimal retrieval
   - Include overlap between chunks for context preservation
   - Store chunk position for result ordering

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

### PgVectorStore
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

## Contributing

This package is part of the Rakam Systems framework. For contributions and development:

1. Follow the existing interface patterns in `ai_core.interfaces`
2. Add comprehensive docstrings and type hints
3. Include examples in the `examples/` directory
4. Update this README with new features

## License

See [LICENSE](LICENSE) file for details.

## Additional Resources

- [Main Rakam Systems Documentation](../../README.md)
- [AI Core Interfaces](../ai_core/interfaces/)
- [Examples Directory](../../examples/ai_vectorstore_examples/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [SentenceTransformers Documentation](https://www.sbert.net/)

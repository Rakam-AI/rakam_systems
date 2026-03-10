# Rakam System Vectorstore

The vectorstore package of Rakam Systems providing vector database solutions and document processing capabilities.

## Overview

`rakam-systems-vectorstore` provides comprehensive vector storage, embedding models, and document loading capabilities. This package depends on `rakam-systems-core`.

## Features

- **Configuration-First Design**: Change your entire vector store setup via YAML - no code changes
- **Multiple Backends**: PostgreSQL with pgvector and FAISS in-memory storage
- **Flexible Embeddings**: Support for SentenceTransformers, OpenAI, and Cohere
- **Document Loaders**: PDF, DOCX, HTML, Markdown, CSV, and more
- **Search Capabilities**: Vector search, keyword search (BM25), and hybrid search
- **Chunking**: Intelligent text chunking with context preservation
- **Configuration**: Comprehensive YAML/JSON configuration support

### Configuration Convenience

The vectorstore package's configurable design allows you to:

- **Switch embedding models** without code changes (local ↔ OpenAI ↔ Cohere)
- **Change search algorithms** instantly (BM25 ↔ ts_rank ↔ hybrid)
- **Adjust search parameters** (similarity metrics, top-k, hybrid weights)
- **Toggle features** (hybrid search, caching, reranking)
- **Tune performance** (batch sizes, chunk sizes, connection pools)
- **Swap backends** (FAISS ↔ PostgreSQL) by updating config

## Installation

```bash
# Install vectorstore package
pip install rakam-systems-vectorstore

# With specific backends
pip install rakam-systems-vectorstore[postgres]
pip install rakam-systems-vectorstore[faiss]
pip install rakam-systems-vectorstore[all]
```

Available extras:

| Extra              | What it adds                                                                     |
| ------------------ | -------------------------------------------------------------------------------- |
| `postgres`         | `psycopg2-binary`, `pgvector`, `django`                                          |
| `faiss`            | `faiss-cpu`                                                                      |
| `local-embeddings` | `sentence-transformers`, `torch`                                                 |
| `openai`           | `openai` (for OpenAI embeddings)                                                 |
| `cohere`           | `cohere` (for Cohere embeddings)                                                 |
| `loaders`          | `python-magic`, `beautifulsoup4`, `python-docx`, `pymupdf`, `docling`, `chonkie` |
| `all`              | Everything above                                                                 |

## Quick Start

### FAISS Vector Store (In-Memory)

```python
from rakam_systems_vectorstore import FaissStore, Node, NodeMetadata

store = FaissStore(
    name="my_store",
    base_index_path="./indexes",
    embedding_model="Snowflake/snowflake-arctic-embed-m",
    initialising=True
)

nodes = [
    Node(
        content="Python is great for AI",
        metadata=NodeMetadata(source_file_uuid="doc1", position=0)
    )
]

store.create_collection_from_nodes("my_collection", nodes)
results, _ = store.search(collection_name="my_collection", query="AI programming", number=5)
for _, (_, content, dist) in results.items():
    print(f"[{dist:.4f}] {content}")
```

### PostgreSQL Vector Store

#### Set up PostgreSQL with pgvector

`ConfigurablePgVectorStore` requires a PostgreSQL instance with the [pgvector](https://github.com/pgvector/pgvector) extension.

Start a local instance with Docker:

```bash
docker run -d --name postgres-vectorstore \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vectorstore_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Then configure the connection via environment variables:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=vectorstore_db
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
```

#### Connect and use

```python
import os
import django
from django.conf import settings

# Configure Django (required)
if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'rakam_systems_vectorstore.components.vectorstore',
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.postgresql',
                'NAME': os.getenv('POSTGRES_DB', 'vectorstore_db'),
                'USER': os.getenv('POSTGRES_USER', 'postgres'),
                'PASSWORD': os.getenv('POSTGRES_PASSWORD', 'postgres'),
                'HOST': os.getenv('POSTGRES_HOST', 'localhost'),
                'PORT': os.getenv('POSTGRES_PORT', '5432'),
            }
        },
        DEFAULT_AUTO_FIELD='django.db.models.BigAutoField',
    )
    django.setup()

from rakam_systems_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

# Create configuration
config = VectorStoreConfig(
    embedding={
        "model_type": "sentence_transformer",
        "model_name": "Snowflake/snowflake-arctic-embed-m"
    },
    search={
        "similarity_metric": "cosine",
        "enable_hybrid_search": True
    }
)

# Create and use store
store = ConfigurablePgVectorStore(config=config)
store.setup()
store.add_nodes(nodes)
results = store.search("What is AI?", top_k=5)
store.shutdown()
```

## Core Components

### Vector Stores

- **ConfigurablePgVectorStore**: PostgreSQL with pgvector, supports hybrid search and keyword search
- **FaissStore**: In-memory FAISS-based vector search

### Embeddings

- **ConfigurableEmbeddings**: Supports multiple backends
  - SentenceTransformers (local)
  - OpenAI embeddings
  - Cohere embeddings

### Document Loaders

- **AdaptiveLoader**: Automatically detects and loads various file types
- **PdfLoader**: Advanced PDF processing with Docling
- **PdfLoaderLight**: Lightweight PDF to markdown conversion
- **DocLoader**: Microsoft Word documents
- **OdtLoader**: OpenDocument Text files
- **MdLoader**: Markdown files
- **HtmlLoader**: HTML files
- **EmlLoader**: Email files
- **TabularLoader**: CSV, Excel files
- **CodeLoader**: Source code files

### Chunking

- **TextChunker**: Sentence-based chunking with Chonkie
- **AdvancedChunker**: Context-aware chunking with heading preservation

## Package Structure

```
rakam-systems-vectorstore/
├── src/rakam_systems_vectorstore/
│   ├── core.py                  # Node, VSFile, NodeMetadata
│   ├── config.py                # VectorStoreConfig
│   ├── components/
│   │   ├── vectorstore/         # Store implementations
│   │   │   ├── configurable_pg_vectorstore.py
│   │   │   └── faiss_vector_store.py
│   │   ├── embedding_model/     # Embedding models
│   │   │   └── configurable_embeddings.py
│   │   ├── loader/              # Document loaders
│   │   │   ├── adaptive_loader.py
│   │   │   ├── pdf_loader.py
│   │   │   ├── pdf_loader_light.py
│   │   │   └── ... (other loaders)
│   │   └── chunker/             # Text chunkers
│   │       ├── text_chunker.py
│   │       └── advanced_chunker.py
│   ├── docs/                    # Package documentation
│   └── server/                  # MCP server
└── pyproject.toml
```

## Search Capabilities

### Vector Search

Semantic similarity search using embeddings:

```python
results = store.search("machine learning algorithms", top_k=10)
```

### Keyword Search (BM25)

Full-text search with BM25 ranking:

```python
results = store.keyword_search(
    query="machine learning",
    top_k=10,
    ranking_algorithm="bm25"
)
```

### Hybrid Search

Combines vector and keyword search:

```python
results = store.hybrid_search(
    query="neural networks",
    top_k=10,
    alpha=0.7  # 70% vector, 30% keyword
)
```

## Configuration

### From YAML

```yaml
# vectorstore_config.yaml
name: my_vectorstore

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
```

```python
config = VectorStoreConfig.from_yaml("vectorstore_config.yaml")
store = ConfigurablePgVectorStore(config=config)
```

## Examples

See the `examples/ai_vectorstore_examples/` directory in the main repository for complete examples:

- Basic FAISS example
- PostgreSQL example
- Configurable vectorstore examples
- PDF loader examples
- Keyword search examples

## Environment Variables

- `POSTGRES_HOST`: PostgreSQL host (default: localhost)
- `POSTGRES_PORT`: PostgreSQL port (default: 5432)
- `POSTGRES_DB`: Database name (default: vectorstore_db)
- `POSTGRES_USER`: Database user (default: postgres)
- `POSTGRES_PASSWORD`: Database password
- `OPENAI_API_KEY`: For OpenAI embeddings
- `COHERE_API_KEY`: For Cohere embeddings
- `HUGGINGFACE_TOKEN`: For private HuggingFace models

## License

Apache 2.0

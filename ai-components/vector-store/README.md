# Rakam System Vectorstore

The vectorstore package of Rakam Systems providing vector database solutions and document processing capabilities.

## Overview

`rakam-systems-vectorstore` provides comprehensive vector storage, embedding models, and document loading capabilities. This package depends on `rakam-systems-core`.

## Features

- **Configuration-First Design**: Change your entire vector store setup via YAML — no code changes
- **Multiple Backends**: PostgreSQL with pgvector and FAISS in-memory storage
- **Flexible Embeddings**: SentenceTransformers, OpenAI, and Cohere
- **Document Loaders**: PDF, DOCX, HTML, Markdown, CSV, and more
- **Search Capabilities**: Vector search, keyword search (BM25), and hybrid search
- **Chunking**: Intelligent text chunking with context preservation

## Installation

```bash
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
```

## Core Components

- **ConfigurablePgVectorStore** — PostgreSQL with pgvector, hybrid search, keyword search
- **FaissStore** — In-memory FAISS-based vector search
- **ConfigurableEmbeddings** — SentenceTransformers, OpenAI, Cohere backends
- **AdaptiveLoader** — Auto-detects and loads PDF, DOCX, HTML, Markdown, CSV, email, code
- **TextChunker / AdvancedChunker** — Sentence-based and context-aware chunking

## Environment Variables

| Variable | Description |
|----------|-------------|
| `POSTGRES_HOST` | PostgreSQL host (default: localhost) |
| `POSTGRES_PORT` | PostgreSQL port (default: 5432) |
| `POSTGRES_DB` | Database name (default: vectorstore_db) |
| `POSTGRES_USER` | Database user (default: postgres) |
| `POSTGRES_PASSWORD` | Database password |
| `OPENAI_API_KEY` | For OpenAI embeddings |
| `COHERE_API_KEY` | For Cohere embeddings |
| `HUGGINGFACE_TOKEN` | For private HuggingFace models |

## Documentation

For PostgreSQL setup, search examples, YAML configuration, and full API reference, see the [official documentation](https://rakam-ai.github.io/rakam-systems-docs/).

## License

Apache 2.0

---
title: Vector Stores
---

# Vector Stores

Vector Store is a standalone component for storing and searching document embeddings. It can be used independently for semantic search, or as the data layer in a RAG pipeline (see [Agents](./agents.md#build-a-rag-pipeline)).

```bash
pip install rakam-systems-vectorstore[all]
```

Available extras:

| Extra | What it adds |
|-------|-------------|
| `postgres` | `psycopg2-binary`, `pgvector`, `django` |
| `faiss` | `faiss-cpu` |
| `local-embeddings` | `sentence-transformers`, `torch` |
| `openai` | `openai` (for OpenAI embeddings) |
| `cohere` | `cohere` (for Cohere embeddings) |
| `loaders` | `python-magic`, `beautifulsoup4`, `python-docx`, `pymupdf`, `docling`, `chonkie` |
| `all` | Everything above |

## Use FAISS (in-memory)

```python
from rakam_systems_vectorstore import FaissStore, Node, NodeMetadata

store = FaissStore(
    name="my_store",
    base_index_path="./indexes",
    embedding_model="Snowflake/snowflake-arctic-embed-m",
    initialising=True
)

nodes = [
    Node(content="Python is a programming language.",
         metadata=NodeMetadata(source_file_uuid="doc1", position=0)),
    Node(content="Machine learning is AI subset.",
         metadata=NodeMetadata(source_file_uuid="doc1", position=1)),
]
store.create_collection_from_nodes("docs", nodes)

results, _ = store.search(collection_name="docs", query="What is Python?", number=3)
for _, (_, content, dist) in results.items():
    print(f"[{dist:.4f}] {content}")
```

## Use PostgreSQL (production)

```python
# Requires: pip install rakam-systems-vectorstore[all]
# Requires: PostgreSQL with pgvector extension

from rakam_systems_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

config = VectorStoreConfig(
    name="prod_store",
    embedding={"model_type": "sentence_transformer", "model_name": "Snowflake/snowflake-arctic-embed-m"}
)
# Database config via environment variables (POSTGRES_HOST, etc.)

store = ConfigurablePgVectorStore(config=config)
store.setup()
# ... use store ...
store.shutdown()
```

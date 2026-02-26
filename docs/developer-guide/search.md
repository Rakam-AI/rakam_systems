---
title: Search
---

# Search

Rakam Systems supports vector search, keyword search, and hybrid search through `ConfigurablePgVectorStore`.

## Django setup

`ConfigurablePgVectorStore` uses Django's ORM for PostgreSQL access. Configure Django before creating a store:

```python
import os
import django
from django.conf import settings

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
```

## Vector search

Semantic similarity search using embeddings:

```python
from rakam_systems_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

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

store = ConfigurablePgVectorStore(config=config)
store.setup()

# Add documents
store.add_nodes(nodes)
store.add_vsfile(vsfile)

# Search
results = store.search("What is machine learning?", top_k=5)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:200]}...")

store.shutdown()
```

### FaissStore

In-memory FAISS-based vector search, useful for development or small datasets that do not require PostgreSQL.

### Multi-model support

Each embedding model automatically gets dedicated tables to prevent mixing incompatible vector spaces:

```python
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

## Keyword search

Full-text search using PostgreSQL's built-in capabilities with BM25 or ts_rank ranking:

```python
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

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:200]}...")
```

### Configure via VectorStoreConfig

```python
config = VectorStoreConfig(
    search={
        "keyword_ranking_algorithm": "bm25",
        "keyword_k1": 1.2,
        "keyword_b": 0.75
    }
)
```

### Ranking algorithms

- **BM25**: Best Match 25, probabilistic ranking function
  - `k1`: Term frequency saturation parameter (default: 1.2)
  - `b`: Length normalization parameter (default: 0.75)
- **ts_rank**: PostgreSQL's text search ranking function
  - Weights different parts of documents differently
  - Good for structured documents

## Hybrid search

Combines vector search and keyword search with a configurable alpha parameter:

```python
# alpha controls the blend: 1.0 = pure vector, 0.0 = pure keyword
results = store.hybrid_search("machine learning", top_k=10, alpha=0.7)
```

Configure hybrid search defaults via VectorStoreConfig:

```python
config = VectorStoreConfig(
    search={
        "enable_hybrid_search": True,
        "hybrid_alpha": 0.7
    }
)
```

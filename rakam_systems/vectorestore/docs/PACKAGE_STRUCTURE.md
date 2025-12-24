# AI Vectorstore - Package Structure

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     rakam_systems Package                        │
│                                                                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │    ai_core      │  │  ai_vectorstore  │  │   ai_agents    │ │
│  │   (required)    │  │   (optional)     │  │   (optional)   │ │
│  │                 │  │                  │  │                │ │
│  │  - interfaces/  │←─│  - core.py       │  │  - components/ │ │
│  │  - base.py      │  │  - config.py     │  │  - server/     │ │
│  │  - tracking.py  │  │  - components/   │  │                │ │
│  └─────────────────┘  └──────────────────┘  └────────────────┘ │
│         ▲                      │                                 │
│         │                      │                                 │
│         └──────────────────────┘                                 │
│              dependency                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Installation Hierarchy

```
Installation Options
│
├─ Full Framework
│  └─ pip install rakam-systems[all]
│     ├─ ai_core (always included)
│     ├─ ai_vectorstore (with all deps)
│     ├─ ai_agents
│     └─ ai_utils
│
├─ Selective Components
│  └─ pip install rakam-systems[ai-vectorstore]
│     ├─ ai_core (always included)
│     └─ ai_vectorstore (with all deps)
│
└─ Standalone Submodule
   └─ pip install ./ai_vectorstore[...]
      ├─ ai_core (interfaces only)
      └─ ai_vectorstore
         │
         ├─ [postgres] → Django + psycopg2
         ├─ [faiss] → faiss-cpu
         ├─ [local-embeddings] → sentence-transformers + torch
         ├─ [openai] → openai
         ├─ [cohere] → cohere
         ├─ [loaders] → document parsers
         └─ [all] → everything above
```

## Dependency Graph

```
ai_vectorstore Components
│
├─ Core (always installed)
│  ├─ core.py (Node, VSFile, NodeMetadata)
│  ├─ config.py (VectorStoreConfig, *)
│  └─ Depends on: pyyaml, numpy, tqdm
│
├─ [postgres] extra
│  ├─ pg_vector_store.py
│  ├─ configurable_pg_vector_store.py
│  ├─ pg_models.py
│  └─ Depends on: django, psycopg2-binary
│
├─ [faiss] extra
│  ├─ faiss_vector_store.py
│  └─ Depends on: faiss-cpu
│
├─ [local-embeddings] extra
│  ├─ configurable_embeddings.py
│  └─ Depends on: sentence-transformers, torch
│
├─ [openai] extra
│  ├─ openai_embeddings.py
│  └─ Depends on: openai
│
├─ [cohere] extra
│  └─ Depends on: cohere
│
└─ [loaders] extra
   ├─ adaptive_loader.py
   ├─ file_loader.py
   └─ Depends on: pymupdf, python-docx, beautifulsoup4, etc.
```

## Import Resolution

```
User Code
   │
   ├─ from rakam_systems.ai_vectorstore import ConfigurablePgVectorStore
   │                  │
   │                  └─→ rakam_systems/ai_vectorstore/__init__.py
   │                                  │
   │                                  └─→ Lazy import from components/
   │
   ├─ from rakam_systems.ai_vectorstore.config import VectorStoreConfig
   │                  │
   │                  └─→ rakam_systems/ai_vectorstore/config.py
   │
   └─ from rakam_systems.ai_core.interfaces.vectorstore import VectorStore
                      │
                      └─→ rakam_systems/ai_core/interfaces/vectorstore.py
```

## File Organization

```
rakam_systems/
│
├── ai_core/                          # Core interfaces (always included)
│   ├── __init__.py
│   ├── base.py                       # BaseComponent
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── vectorstore.py           # VectorStore interface
│   │   ├── embedding_model.py       # EmbeddingModel interface
│   │   ├── chunker.py               # Chunker interface
│   │   ├── loader.py                # Loader interface
│   │   ├── retriever.py             # Retriever interface
│   │   └── reranker.py              # Reranker interface
│   └── ...
│
└── ai_vectorstore/                   # Standalone submodule
    ├── __init__.py                   # Public API exports
    ├── setup.py                      # Standalone installation
    ├── pyproject.toml                # Package configuration
    ├── MANIFEST.in                   # Package manifest
    │
    ├── README.md                     # Main documentation
    ├── INSTALLATION.md               # Installation guide
    ├── QUICK_INSTALL.md              # Quick reference
    ├── ARCHITECTURE.md               # Design decisions
    ├── PACKAGE_STRUCTURE.md          # This file
    │
    ├── core.py                       # Core data structures
    ├── config.py                     # Configuration system
    │
    ├── components/                   # Implementation components
    │   ├── __init__.py
    │   │
    │   ├── vectorstore/              # Vector store backends
    │   │   ├── __init__.py
    │   │   ├── pg_vector_store.py          # PostgreSQL (legacy)
    │   │   ├── configurable_pg_vector_store.py  # PostgreSQL (new)
    │   │   ├── faiss_vector_store.py       # FAISS backend
    │   │   ├── pg_models.py                # Django models
    │   │   └── ...
    │   │
    │   ├── embedding_model/          # Embedding implementations
    │   │   ├── __init__.py
    │   │   ├── configurable_embeddings.py  # Multi-backend
    │   │   ├── openai_embeddings.py        # OpenAI API
    │   │   └── ...
    │   │
    │   ├── chunker/                  # Text chunking
    │   │   ├── __init__.py
    │   │   ├── simple_chunker.py
    │   │   └── text_chunker.py
    │   │
    │   ├── loader/                   # Document loading
    │   │   ├── __init__.py
    │   │   ├── file_loader.py
    │   │   └── adaptive_loader.py
    │   │
    │   ├── indexer/                  # Indexing
    │   │   ├── __init__.py
    │   │   └── simple_indexer.py
    │   │
    │   ├── retriever/                # Retrieval
    │   │   ├── __init__.py
    │   │   └── basic_retriever.py
    │   │
    │   └── reranker/                 # Re-ranking
    │       ├── __init__.py
    │       └── model_reranker.py
    │
    └── server/                       # MCP server
        ├── __init__.py
        └── mcp_server_vector.py
```

## Class Hierarchy

```
BaseComponent (from ai_core.base)
│
├── VectorStore (interface from ai_core.interfaces.vectorstore)
│   ├── PgVectorStore (legacy implementation)
│   ├── ConfigurablePgVectorStore (new implementation) ⭐
│   └── FaissStore (FAISS implementation)
│
├── EmbeddingModel (interface from ai_core.interfaces.embedding_model)
│   ├── ConfigurableEmbeddings (multi-backend) ⭐
│   └── OpenAIEmbeddings (OpenAI API)
│
├── Chunker (interface from ai_core.interfaces.chunker)
│   ├── SimpleChunker
│   └── TextChunker
│
├── Loader (interface from ai_core.interfaces.loader)
│   ├── FileLoader
│   └── AdaptiveLoader ⭐
│
├── Retriever (interface from ai_core.interfaces.retriever)
│   └── BasicRetriever
│
├── Reranker (interface from ai_core.interfaces.reranker)
│   └── ModelReranker
│
└── Indexer (interface from ai_core.interfaces.indexer)
    └── SimpleIndexer

⭐ = Recommended for new projects
```

## Data Flow

```
User Application
      │
      ├──→ Load Documents
      │         │
      │         ↓
      │    [Loader] → VSFile[]
      │         │
      │         ↓
      │    [Chunker] → Node[]
      │         │
      │         ↓
      │    [EmbeddingModel] → embeddings[]
      │         │
      │         ↓
      │    [VectorStore.add_nodes()]
      │         │
      │         ↓
      │    PostgreSQL or FAISS
      │
      └──→ Search Query
                │
                ↓
           [EmbeddingModel] → query_embedding
                │
                ↓
           [VectorStore.search()]
                │
                ├──→ Vector Search
                ├──→ Hybrid Search (optional)
                └──→ [Reranker] (optional)
                │
                ↓
           [Retriever] → Node[]
                │
                ↓
           Results to User
```

## Configuration Flow

```
User Configuration
      │
      ├─ YAML File
      ├─ JSON File
      └─ Python Dict
      │
      ↓
VectorStoreConfig.from_yaml/from_json/from_dict()
      │
      ├─→ EmbeddingConfig
      │     ├─ model_type
      │     ├─ model_name
      │     ├─ api_key (from env)
      │     └─ ...
      │
      ├─→ DatabaseConfig
      │     ├─ host (from env)
      │     ├─ port (from env)
      │     └─ ...
      │
      ├─→ SearchConfig
      │     ├─ similarity_metric
      │     ├─ hybrid_alpha
      │     └─ ...
      │
      └─→ IndexConfig
            ├─ chunk_size
            ├─ batch_insert_size
            └─ ...
      │
      ↓
ConfigurablePgVectorStore(config=config)
      │
      └─→ Components configured automatically
```

## Package Dependencies

```
Core Dependencies (always installed):
  - pyyaml >= 6.0
  - numpy >= 1.24.0
  - tqdm >= 4.66.0

Optional Dependencies:

  [postgres]:
    - psycopg2-binary >= 2.9.9
    - django >= 4.0.0

  [faiss]:
    - faiss-cpu >= 1.12.0

  [local-embeddings]:
    - sentence-transformers >= 5.1.0
    - torch >= 2.0.0

  [openai]:
    - openai >= 1.0.0

  [cohere]:
    - cohere >= 4.0.0

  [loaders]:
    - python-magic >= 0.4.27
    - beautifulsoup4 >= 4.12.0
    - python-docx >= 1.2.0
    - pymupdf >= 1.24.0
    - pymupdf4llm >= 0.0.17

Interface Dependencies (included in standalone):
  - rakam_systems.ai_core.interfaces.*
  - rakam_systems.ai_core.base
```

## Installation Comparison

### Scenario 1: Minimal Development Setup

```bash
# Install FAISS + local embeddings only
cd rakam_systems/ai_vectorstore
pip install -e ".[faiss,local-embeddings]"

# Results in:
✓ ai_core interfaces
✓ ai_vectorstore core
✓ faiss-cpu
✓ sentence-transformers
✓ torch
✗ Django (not installed)
✗ PostgreSQL (not needed)
```

### Scenario 2: Production with PostgreSQL

```bash
# Install via main framework
cd rakam_systems
pip install -e ".[ai-vectorstore]"

# Results in:
✓ ai_core (full)
✓ ai_vectorstore (all components)
✓ PostgreSQL + Django
✓ FAISS
✓ SentenceTransformers
✓ OpenAI + Cohere
✓ All loaders
```

### Scenario 3: Lightweight API-Based

```bash
# Install with OpenAI embeddings only
cd rakam_systems/ai_vectorstore
pip install -e ".[faiss,openai]"

# Results in:
✓ ai_core interfaces
✓ ai_vectorstore core
✓ faiss-cpu
✓ openai
✗ sentence-transformers (not needed)
✗ torch (not needed)
```

## Public API Surface

### Main Exports (`__init__.py`)

```python
from rakam_systems.ai_vectorstore import (
    # Data structures
    Node,
    NodeMetadata,
    VSFile,
    
    # Configuration
    VectorStoreConfig,
    EmbeddingConfig,
    DatabaseConfig,
    SearchConfig,
    IndexConfig,
    load_config,
    
    # Vector stores (lazy loaded)
    ConfigurablePgVectorStore,  # ⭐ Recommended
    PgVectorStore,              # Legacy
    FaissVectorStore,
    
    # Components (lazy loaded)
    AdaptiveLoader,
    ConfigurableEmbeddings,
    create_embedding_model,
    create_adaptive_loader,
)
```

### Configuration API

```python
from rakam_systems.ai_vectorstore.config import (
    VectorStoreConfig,
    EmbeddingConfig,
    DatabaseConfig,
    SearchConfig,
    IndexConfig,
    load_config,
)
```

### Component APIs

```python
# Vector stores
from rakam_systems.ai_vectorstore.components.vectorstore import (
    ConfigurablePgVectorStore,
    PgVectorStore,
    FaissStore,
)

# Embeddings
from rakam_systems.ai_vectorstore.components.embedding_model import (
    ConfigurableEmbeddings,
    OpenAIEmbeddings,
)

# Loaders
from rakam_systems.ai_vectorstore.components.loader import (
    AdaptiveLoader,
    FileLoader,
)

# Other components...
```

## Summary

The `ai_vectorstore` package:

✅ **Modular**: Install only what you need  
✅ **Standalone**: Can work independently  
✅ **Integrated**: Works with other rakam_systems components  
✅ **Configurable**: YAML/JSON configuration support  
✅ **Extensible**: Clear interfaces for custom implementations  
✅ **Production-Ready**: Multiple backend options  

Import as:
```python
from rakam_systems.ai_vectorstore import ConfigurablePgVectorStore
```

Whether installed via:
- `pip install rakam-systems[ai-vectorstore]`
- `pip install ./ai_vectorstore[all]`
- `pip install ./ai_vectorstore[postgres,local-embeddings]`

---

**See also**:
- [INSTALLATION.md](INSTALLATION.md) - Installation guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions
- [README.md](README.md) - Usage documentation


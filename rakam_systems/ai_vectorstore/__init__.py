"""
AI Vector Store Package

Provides modular, configurable vector storage and retrieval components
leveraging PGVector for efficient semantic search.

Key Components:
- ConfigurablePgVectorStore: Enhanced vector store with full configuration support
- AdaptiveLoader: Automatically detects and processes various file types
- ConfigurableEmbeddings: Multi-backend embedding model support
- VectorStoreConfig: Comprehensive configuration management

Quick Start:
    from ai_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig
    
    config = VectorStoreConfig()
    store = ConfigurablePgVectorStore(config=config)
    store.setup()
"""

from ai_vectorstore.core import Node, NodeMetadata, VSFile
from ai_vectorstore.config import (
    VectorStoreConfig,
    EmbeddingConfig,
    DatabaseConfig,
    SearchConfig,
    IndexConfig,
    load_config,
)

# Import new configurable components
from ai_vectorstore.components.vectorstore.configurable_pg_vector_store import (
    ConfigurablePgVectorStore,
)
from ai_vectorstore.components.loader.adaptive_loader import (
    AdaptiveLoader,
    create_adaptive_loader,
)
from ai_vectorstore.components.embedding_model.configurable_embeddings import (
    ConfigurableEmbeddings,
    create_embedding_model,
)

# Import original components for backward compatibility
from ai_vectorstore.components.vectorstore.pg_vector_store import PgVectorStore
from ai_vectorstore.components.vectorstore.faiss_vector_store import FaissStore as FaissVectorStore

__version__ = "1.0.0"

__all__ = [
    # Core data structures
    "Node",
    "NodeMetadata",
    "VSFile",
    
    # Configuration
    "VectorStoreConfig",
    "EmbeddingConfig",
    "DatabaseConfig",
    "SearchConfig",
    "IndexConfig",
    "load_config",
    
    # New configurable components
    "ConfigurablePgVectorStore",
    "AdaptiveLoader",
    "create_adaptive_loader",
    "ConfigurableEmbeddings",
    "create_embedding_model",
    
    # Original components (backward compatibility)
    "PgVectorStore",
    "FaissVectorStore",
]


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

from rakam_systems.ai_vectorstore.core import Node, NodeMetadata, VSFile
from rakam_systems.ai_vectorstore.config import (
    VectorStoreConfig,
    EmbeddingConfig,
    DatabaseConfig,
    SearchConfig,
    IndexConfig,
    load_config,
)

# Lazy imports for Django model-dependent components
# These will be imported when accessed, after Django setup
def __getattr__(name):
    """Lazy import Django-dependent components."""
    if name == "ConfigurablePgVectorStore":
        from rakam_systems.ai_vectorstore.components.vectorstore.configurable_pg_vector_store import (
            ConfigurablePgVectorStore,
        )
        return ConfigurablePgVectorStore
    elif name == "PgVectorStore":
        from rakam_systems.ai_vectorstore.components.vectorstore.pg_vector_store import PgVectorStore
        return PgVectorStore
    elif name == "AdaptiveLoader":
        from rakam_systems.ai_vectorstore.components.loader.adaptive_loader import AdaptiveLoader
        return AdaptiveLoader
    elif name == "create_adaptive_loader":
        from rakam_systems.ai_vectorstore.components.loader.adaptive_loader import create_adaptive_loader
        return create_adaptive_loader
    elif name == "ConfigurableEmbeddings":
        from rakam_systems.ai_vectorstore.components.embedding_model.configurable_embeddings import (
            ConfigurableEmbeddings,
        )
        return ConfigurableEmbeddings
    elif name == "create_embedding_model":
        from rakam_systems.ai_vectorstore.components.embedding_model.configurable_embeddings import (
            create_embedding_model,
        )
        return create_embedding_model
    elif name == "FaissVectorStore":
        from rakam_systems.ai_vectorstore.components.vectorstore.faiss_vector_store import FaissStore as FaissVectorStore
        return FaissVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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


"""
Configuration management for Vector Store components.

This module provides a unified configuration system that supports:
- YAML/JSON configuration files
- Environment variable overrides
- Programmatic configuration
- Validation and defaults
"""

from __future__ import annotations

import os
import yaml
import json
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    
    model_type: str = "sentence_transformer"  # sentence_transformer, openai, cohere, etc.
    model_name: str = "Snowflake/snowflake-arctic-embed-m"
    api_key: Optional[str] = None
    batch_size: int = 32
    normalize: bool = True
    dimensions: Optional[int] = None  # Auto-detected if None
    
    def __post_init__(self):
        # Load API key from environment if not provided
        if self.model_type == "openai" and not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.model_type == "cohere" and not self.api_key:
            self.api_key = os.getenv("COHERE_API_KEY")


@dataclass
class DatabaseConfig:
    """Configuration for database connection."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "vectorstore_db"
    user: str = "postgres"
    password: str = "postgres"
    pool_size: int = 10
    max_overflow: int = 20
    
    def __post_init__(self):
        # Load from environment variables if available
        self.host = os.getenv("POSTGRES_HOST", self.host)
        self.port = int(os.getenv("POSTGRES_PORT", str(self.port)))
        self.database = os.getenv("POSTGRES_DB", self.database)
        self.user = os.getenv("POSTGRES_USER", self.user)
        self.password = os.getenv("POSTGRES_PASSWORD", self.password)
    
    def to_connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    
    similarity_metric: str = "cosine"  # cosine, l2, dot_product
    default_top_k: int = 5
    enable_hybrid_search: bool = True
    hybrid_alpha: float = 0.7  # Weight for vector similarity (1-alpha for keyword)
    rerank: bool = True
    search_buffer_factor: int = 2  # Retrieve more results for reranking
    
    def validate(self):
        """Validate search configuration."""
        if self.similarity_metric not in ["cosine", "l2", "dot_product", "dot"]:
            raise ValueError(f"Invalid similarity metric: {self.similarity_metric}")
        if not 0 <= self.hybrid_alpha <= 1:
            raise ValueError(f"hybrid_alpha must be between 0 and 1, got {self.hybrid_alpha}")
        if self.default_top_k < 1:
            raise ValueError(f"default_top_k must be >= 1, got {self.default_top_k}")


@dataclass
class IndexConfig:
    """Configuration for indexing operations."""
    
    chunk_size: int = 512
    chunk_overlap: int = 50
    enable_parallel_processing: bool = False
    parallel_workers: int = 4
    batch_insert_size: int = 100


@dataclass
class VectorStoreConfig:
    """Master configuration for Vector Store component."""
    
    name: str = "pg_vector_store"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    
    # Component-specific settings
    enable_caching: bool = True
    cache_size: int = 1000
    enable_logging: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorStoreConfig":
        """Create configuration from dictionary."""
        # Extract nested configs
        embedding_config = EmbeddingConfig(**config_dict.get("embedding", {}))
        database_config = DatabaseConfig(**config_dict.get("database", {}))
        search_config = SearchConfig(**config_dict.get("search", {}))
        index_config = IndexConfig(**config_dict.get("index", {}))
        
        # Create main config
        main_config = {
            "name": config_dict.get("name", "pg_vector_store"),
            "embedding": embedding_config,
            "database": database_config,
            "search": search_config,
            "index": index_config,
            "enable_caching": config_dict.get("enable_caching", True),
            "cache_size": config_dict.get("cache_size", 1000),
            "enable_logging": config_dict.get("enable_logging", True),
            "log_level": config_dict.get("log_level", "INFO"),
        }
        
        return cls(**main_config)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "VectorStoreConfig":
        """Load configuration from YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "VectorStoreConfig":
        """Load configuration from JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "embedding": asdict(self.embedding),
            "database": asdict(self.database),
            "search": asdict(self.search),
            "index": asdict(self.index),
            "enable_caching": self.enable_caching,
            "cache_size": self.cache_size,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
        }
    
    def save_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> None:
        """Validate all configuration settings."""
        self.search.validate()
        
        # Validate embedding config
        if self.embedding.batch_size < 1:
            raise ValueError(f"embedding.batch_size must be >= 1, got {self.embedding.batch_size}")
        
        # Validate index config
        if self.index.chunk_size < 1:
            raise ValueError(f"index.chunk_size must be >= 1, got {self.index.chunk_size}")
        if self.index.chunk_overlap < 0:
            raise ValueError(f"index.chunk_overlap must be >= 0, got {self.index.chunk_overlap}")
        if self.index.chunk_overlap >= self.index.chunk_size:
            raise ValueError("index.chunk_overlap must be less than index.chunk_size")


def load_config(
    config_source: Optional[Union[str, Path, Dict[str, Any]]] = None,
    config_type: str = "auto"
) -> VectorStoreConfig:
    """
    Load configuration from various sources.
    
    Args:
        config_source: Path to config file, dict, or None for defaults
        config_type: Type of config file ('yaml', 'json', 'auto')
    
    Returns:
        VectorStoreConfig instance
    """
    if config_source is None:
        # Return default configuration
        return VectorStoreConfig()
    
    if isinstance(config_source, dict):
        # Load from dictionary
        return VectorStoreConfig.from_dict(config_source)
    
    # Load from file
    path = Path(config_source)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_source}")
    
    if config_type == "auto":
        # Auto-detect based on file extension
        suffix = path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            config_type = 'yaml'
        elif suffix == '.json':
            config_type = 'json'
        else:
            raise ValueError(f"Cannot auto-detect config type for file: {config_source}")
    
    if config_type == 'yaml':
        return VectorStoreConfig.from_yaml(path)
    elif config_type == 'json':
        return VectorStoreConfig.from_json(path)
    else:
        raise ValueError(f"Unsupported config type: {config_type}")


__all__ = [
    "EmbeddingConfig",
    "DatabaseConfig", 
    "SearchConfig",
    "IndexConfig",
    "VectorStoreConfig",
    "load_config",
]


"""
Configurable embedding model with support for multiple backends.

Supports:
- Sentence Transformers (local models)
- OpenAI API
- Cohere API
- Custom embedding providers
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np

from ai_core.interfaces.embedding_model import EmbeddingModel
from ai_vectorstore.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class ConfigurableEmbeddings(EmbeddingModel):
    """
    Configurable embedding model that supports multiple backends.
    
    This component automatically selects the appropriate embedding backend
    based on configuration and provides a unified interface.
    """
    
    def __init__(
        self,
        name: str = "configurable_embeddings",
        config: Optional[Union[EmbeddingConfig, dict]] = None
    ):
        """
        Initialize configurable embeddings.
        
        Args:
            name: Component name
            config: EmbeddingConfig or dict with embedding settings
        """
        # Parse config first
        if isinstance(config, dict):
            self.embedding_config = EmbeddingConfig(**config)
            config_dict = config
        elif isinstance(config, EmbeddingConfig):
            self.embedding_config = config
            # Convert EmbeddingConfig to dict for parent class
            from dataclasses import asdict
            config_dict = asdict(config)
        else:
            self.embedding_config = EmbeddingConfig()
            config_dict = None
        
        # Pass dict to parent class
        super().__init__(name=name, config=config_dict)
        
        self.model_type = self.embedding_config.model_type
        self.model_name = self.embedding_config.model_name
        self.batch_size = self.embedding_config.batch_size
        self.normalize = self.embedding_config.normalize
        
        # Backend-specific attributes
        self._model = None
        self._client = None
        self._embedding_dim = None
    
    def setup(self) -> None:
        """Initialize the embedding backend."""
        logger.info(f"Setting up {self.model_type} embedding model: {self.model_name}")
        
        if self.model_type == "sentence_transformer":
            self._setup_sentence_transformer()
        elif self.model_type == "openai":
            self._setup_openai()
        elif self.model_type == "cohere":
            self._setup_cohere()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Detect embedding dimension
        if self.embedding_config.dimensions:
            self._embedding_dim = self.embedding_config.dimensions
        else:
            self._embedding_dim = self._detect_embedding_dimension()
        
        logger.info(f"Embedding model initialized with dimension: {self._embedding_dim}")
        super().setup()
    
    def _setup_sentence_transformer(self) -> None:
        """Setup Sentence Transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for sentence_transformer model type. "
                "Install it with: pip install sentence-transformers"
            )
        
        self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
    
    def _setup_openai(self) -> None:
        """Setup OpenAI API client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for openai model type. "
                "Install it with: pip install openai"
            )
        
        api_key = self.embedding_config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide it in config."
            )
        
        self._client = OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model: {self.model_name}")
    
    def _setup_cohere(self) -> None:
        """Setup Cohere API client."""
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere is required for cohere model type. "
                "Install it with: pip install cohere"
            )
        
        api_key = self.embedding_config.api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key not found. Set COHERE_API_KEY environment variable "
                "or provide it in config."
            )
        
        self._client = cohere.Client(api_key)
        logger.info(f"Initialized Cohere client with model: {self.model_name}")
    
    def _detect_embedding_dimension(self) -> int:
        """Detect embedding dimension by encoding a sample text."""
        sample_embedding = self._encode_batch(["sample text for dimension detection"])[0]
        return len(sample_embedding)
    
    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a batch of texts using the configured backend.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of embedding vectors
        """
        if self.model_type == "sentence_transformer":
            return self._encode_sentence_transformer(texts)
        elif self.model_type == "openai":
            return self._encode_openai(texts)
        elif self.model_type == "cohere":
            return self._encode_cohere(texts)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _encode_sentence_transformer(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using Sentence Transformer."""
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_tensor=False,
            normalize_embeddings=False  # We'll normalize separately if needed
        )
        
        # Convert to list of lists
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        return embeddings
    
    def _encode_openai(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using OpenAI API."""
        all_embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error encoding batch with OpenAI: {e}")
                raise
        
        return all_embeddings
    
    def _encode_cohere(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using Cohere API."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self._client.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type="search_document"
                )
                all_embeddings.extend(response.embeddings)
            except Exception as e:
                logger.error(f"Error encoding batch with Cohere: {e}")
                raise
        
        return all_embeddings
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length."""
        embeddings_array = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings_array / norms
        return normalized.tolist()
    
    def run(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        start_time = time.time()
        logger.debug(f"Encoding {len(texts)} texts with {self.model_type}")
        
        # Encode texts
        embeddings = self._encode_batch(texts)
        
        # Normalize if configured
        if self.normalize:
            embeddings = self._normalize_embeddings(embeddings)
        
        elapsed = time.time() - start_time
        logger.debug(f"Encoded {len(texts)} texts in {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/s)")
        
        return embeddings
    
    def encode_query(self, query: str) -> List[float]:
        """
        Encode a single query text.
        
        Args:
            query: Query text to encode
            
        Returns:
            Embedding vector
        """
        embeddings = self.run([query])
        return embeddings[0] if embeddings else []
    
    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Encode multiple documents.
        
        Args:
            documents: List of documents to encode
            
        Returns:
            List of embedding vectors
        """
        return self.run(documents)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if not self.initialized:
            self.setup()
        return self._embedding_dim
    
    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info(f"Shutting down {self.model_type} embedding model")
        
        if self.model_type == "sentence_transformer" and self._model:
            # Clean up CUDA memory if using GPU
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        
        self._model = None
        self._client = None
        super().shutdown()


# Convenience factory function
def create_embedding_model(
    model_type: str = "sentence_transformer",
    model_name: Optional[str] = None,
    **kwargs
) -> ConfigurableEmbeddings:
    """
    Factory function to create an embedding model.
    
    Args:
        model_type: Type of model (sentence_transformer, openai, cohere)
        model_name: Model name/identifier
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured embedding model
    """
    config = EmbeddingConfig(
        model_type=model_type,
        model_name=model_name or _get_default_model_name(model_type),
        **kwargs
    )
    
    return ConfigurableEmbeddings(config=config)


def _get_default_model_name(model_type: str) -> str:
    """Get default model name for a given type."""
    defaults = {
        "sentence_transformer": "Snowflake/snowflake-arctic-embed-m",
        "openai": "text-embedding-3-small",
        "cohere": "embed-english-v3.0"
    }
    return defaults.get(model_type, "")


__all__ = ["ConfigurableEmbeddings", "create_embedding_model"]


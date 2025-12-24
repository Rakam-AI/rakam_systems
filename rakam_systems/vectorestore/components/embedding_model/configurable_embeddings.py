"""
Configurable embedding model with support for multiple backends.

Supports:
- Sentence Transformers (local models)
- OpenAI API
- Cohere API
- Custom embedding providers
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np

from rakam_systems.core.ai_utils import logging
from rakam_systems.core.ai_core.interfaces.embedding_model import EmbeddingModel
from vectorestore.config import EmbeddingConfig

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
        # Skip if already initialized to avoid reloading the model
        if self.initialized:
            logger.debug(
                f"Embedding model {self.model_name} already initialized, skipping setup")
            return

        logger.info(
            f"Setting up {self.model_type} embedding model: {self.model_name}")

        if self.model_type == "sentence_transformer":
            self._setup_sentence_transformer()
        elif self.model_type == "openai":
            pass  # OpenAI client is created on-demand in _encode_openai
        elif self.model_type == "cohere":
            self._setup_cohere()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Detect embedding dimension
        if self.embedding_config.dimensions:
            self._embedding_dim = self.embedding_config.dimensions
        else:
            self._embedding_dim = self._detect_embedding_dimension()

        logger.info(
            f"Embedding model initialized with dimension: {self._embedding_dim}")
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

        # Authenticate with Hugging Face if token is available
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
                logger.info("Successfully authenticated with Hugging Face")
            except ImportError:
                logger.warning(
                    "huggingface-hub is not installed. "
                    "Install it with: pip install huggingface-hub"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to authenticate with Hugging Face: {e}")
        else:
            logger.debug(
                "No HF_TOKEN found in environment, skipping Hugging Face authentication")

        self._model = SentenceTransformer(
            self.model_name, trust_remote_code=True)
        logger.info(f"Loaded SentenceTransformer model: {self.model_name}")

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
        sample_embedding = self._encode_batch(
            ["sample text for dimension detection"])[0]
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
        import gc

        # CRITICAL: Disable tokenizer parallelism to prevent deadlocks in Docker/multiprocessing environments
        # This is a known issue with HuggingFace tokenizers in containerized environments
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        logger.info(
            f"_encode_sentence_transformer() called with {len(texts)} texts")

        # Ensure all texts are strings (sentence_transformers v3.x compatibility)
        sanitized_texts = [str(t) if not isinstance(
            t, str) else t for t in texts]

        total_texts = len(sanitized_texts)
        num_batches = (total_texts + self.batch_size - 1) // self.batch_size

        logger.info(
            f"Total texts: {total_texts}, Batch size: {self.batch_size}, Num batches: {num_batches}")

        if total_texts > 1000:
            logger.info(
                f"Encoding {total_texts} texts with sentence transformer (batch_size={self.batch_size}, {num_batches} batches)...")

        # For large datasets, encode in chunks with progress logging
        if total_texts > 10000:
            all_embeddings = []
            # Log 1000 times during encoding
            log_interval = max(1, num_batches // 1000)
            batch_start_time = time.time()

            logger.info(
                f"Large dataset: processing {total_texts} texts in {num_batches} batches (logging every {log_interval} batches)")

            for i in range(0, total_texts, self.batch_size):
                batch_num = i // self.batch_size + 1
                batch_texts = sanitized_texts[i:i + self.batch_size]

                # Log before encoding starts (for first batch and every log_interval)
                if batch_num == 1 or batch_num % log_interval == 0 or batch_num == num_batches:
                    logger.info(
                        f"Starting batch {batch_num}/{num_batches}: encoding {len(batch_texts)} texts...")
                    batch_encode_start = time.time()

                batch_embeddings = self._model.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,  # Disabled to prevent blocking in Docker/non-TTY environments
                    convert_to_tensor=False,
                    normalize_embeddings=False
                )

                if batch_num == 1 or batch_num % log_interval == 0 or batch_num == num_batches:
                    batch_encode_time = time.time() - batch_encode_start
                    logger.info(
                        f"Batch {batch_num}/{num_batches} encoding completed in {batch_encode_time:.2f}s")

                all_embeddings.append(batch_embeddings)

                # MEMORY OPTIMIZATION: Clear batch_texts reference
                del batch_texts

                if batch_num % log_interval == 0 or batch_num == num_batches:
                    progress_pct = (batch_num / num_batches) * 100
                    elapsed = time.time() - batch_start_time
                    texts_processed = min(i + self.batch_size, total_texts)
                    rate = texts_processed / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_texts - texts_processed) / \
                        rate if rate > 0 else 0
                    logger.info(
                        f"Embedding progress: {batch_num}/{num_batches} batches ({progress_pct:.1f}%) - {texts_processed}/{total_texts} texts - {rate:.1f} texts/s - ETA: {eta_seconds:.0f}s")

            embeddings = np.vstack(all_embeddings)
            # MEMORY OPTIMIZATION: Clear intermediate arrays
            del all_embeddings
        elif num_batches > 1:
            # For medium datasets (multiple batches but < 10000 texts), also log progress
            all_embeddings = []
            batch_start_time = time.time()
            logger.info(
                f"Processing {total_texts} texts in {num_batches} batches...")

            for i in range(0, total_texts, self.batch_size):
                batch_num = i // self.batch_size + 1
                batch_texts = sanitized_texts[i:i + self.batch_size]

                # Log before encoding starts
                logger.info(
                    f"Starting batch {batch_num}/{num_batches}: encoding {len(batch_texts)} texts...")
                batch_encode_start = time.time()

                batch_embeddings = self._model.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,  # Disabled to prevent blocking in Docker/non-TTY environments
                    convert_to_tensor=False,
                    normalize_embeddings=False
                )

                batch_encode_time = time.time() - batch_encode_start
                logger.info(
                    f"Batch {batch_num}/{num_batches} encoding completed in {batch_encode_time:.2f}s")

                all_embeddings.append(batch_embeddings)

                # MEMORY OPTIMIZATION: Clear batch_texts reference
                del batch_texts

                # Log progress for each batch
                progress_pct = (batch_num / num_batches) * 100
                elapsed = time.time() - batch_start_time
                texts_processed = min(i + self.batch_size, total_texts)
                rate = texts_processed / elapsed if elapsed > 0 else 0
                eta_seconds = (total_texts - texts_processed) / \
                    rate if rate > 0 else 0
                logger.info(
                    f"Embedding progress: batch {batch_num}/{num_batches} ({progress_pct:.1f}%) - {texts_processed}/{total_texts} texts - {rate:.1f} texts/s - ETA: {eta_seconds:.0f}s")

            embeddings = np.vstack(all_embeddings)
            # MEMORY OPTIMIZATION: Clear intermediate arrays
            del all_embeddings
        else:
            # Single batch - but still process in smaller chunks to show progress
            logger.info(
                f"Processing {total_texts} texts (will process in mini-batches of {self.batch_size})...")
            all_embeddings = []
            batch_start_time = time.time()

            # Process in mini-batches even for "single batch" to show progress
            # Use smaller batches for better progress visibility
            mini_batch_size = min(self.batch_size, 32)
            num_mini_batches = (
                total_texts + mini_batch_size - 1) // mini_batch_size

            logger.info(
                f"Will process {total_texts} texts in {num_mini_batches} mini-batches of size {mini_batch_size}")

            for i in range(0, total_texts, mini_batch_size):
                batch_num = i // mini_batch_size + 1
                batch_texts = sanitized_texts[i:i + mini_batch_size]

                # Log before encoding starts
                logger.info(
                    f"Starting mini-batch {batch_num}/{num_mini_batches}: encoding {len(batch_texts)} texts...")
                batch_encode_start = time.time()

                batch_embeddings = self._model.encode(
                    batch_texts,
                    batch_size=mini_batch_size,
                    show_progress_bar=False,  # Disabled to prevent blocking in Docker/non-TTY environments
                    convert_to_tensor=False,
                    normalize_embeddings=False
                )

                batch_encode_time = time.time() - batch_encode_start
                logger.info(
                    f"Mini-batch {batch_num}/{num_mini_batches} encoding completed in {batch_encode_time:.2f}s")

                all_embeddings.append(batch_embeddings)

                # MEMORY OPTIMIZATION: Clear batch_texts reference
                del batch_texts

                # Log progress for each mini-batch
                progress_pct = (batch_num / num_mini_batches) * 100
                elapsed = time.time() - batch_start_time
                texts_processed = min(i + mini_batch_size, total_texts)
                rate = texts_processed / elapsed if elapsed > 0 else 0
                eta_seconds = (total_texts - texts_processed) / \
                    rate if rate > 0 else 0
                logger.info(
                    f"Embedding progress: mini-batch {batch_num}/{num_mini_batches} ({progress_pct:.1f}%) - {texts_processed}/{total_texts} texts - {rate:.1f} texts/s - ETA: {eta_seconds:.0f}s")

            embeddings = np.vstack(all_embeddings)
            # MEMORY OPTIMIZATION: Clear intermediate arrays
            del all_embeddings

        # MEMORY OPTIMIZATION: Clear sanitized_texts as no longer needed
        del sanitized_texts

        # Force garbage collection after encoding to prevent memory buildup
        # This is especially important for long-running batch processes
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Always log completion for visibility
        logger.info(f"✓ Encoding completed for {total_texts} texts")

        # Convert to list of lists and release numpy array
        if isinstance(embeddings, np.ndarray):
            result = embeddings.tolist()
            del embeddings  # Release the numpy array
            gc.collect()
            return result

        return embeddings

    def _encode_openai(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using OpenAI API."""
        from vectorestore.components.embedding_model.openai_embeddings import OpenAIEmbeddings

        # Use the OpenAIEmbeddings implementation with configured batch_size
        openai_embeddings = OpenAIEmbeddings(
            model=self.model_name,
            api_key=self.embedding_config.api_key,
            batch_size=self.batch_size  # Pass the batch_size from ConfigurableEmbeddings
        )
        logger.info(f"OpenAI embeddings using batch_size={self.batch_size}")
        return openai_embeddings.run(texts)

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
        logger.info(
            f"run() called: Encoding {len(texts)} texts with {self.model_type} model '{self.model_name}'")

        # Encode texts
        logger.info(f"Calling _encode_batch() for {len(texts)} texts...")
        embeddings = self._encode_batch(texts)
        logger.info(f"_encode_batch() returned {len(embeddings)} embeddings")

        # Normalize if configured
        if self.normalize:
            logger.info(f"Normalizing {len(embeddings)} embeddings...")
            embeddings = self._normalize_embeddings(embeddings)
            logger.info(f"Normalization complete")

        elapsed = time.time() - start_time
        logger.info(
            f"run() completed: Encoded {len(texts)} texts in {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/s)")

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

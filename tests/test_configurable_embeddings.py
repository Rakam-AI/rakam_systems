"""
Unit tests for Configurable Embeddings.
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock

from rakam_systems.ai_vectorstore.components.embedding_model.configurable_embeddings import (
    ConfigurableEmbeddings,
    create_embedding_model,
)
from rakam_systems.ai_vectorstore.config import EmbeddingConfig


class TestConfigurableEmbeddings(unittest.TestCase):
    """Test ConfigurableEmbeddings class."""
    
    def test_initialization_with_dict_config(self):
        """Test initialization with dict config."""
        config = {
            "model_type": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 16
        }
        
        embedder = ConfigurableEmbeddings(config=config)
        
        self.assertEqual(embedder.model_type, "sentence_transformer")
        self.assertEqual(embedder.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(embedder.batch_size, 16)
    
    def test_initialization_with_embedding_config(self):
        """Test initialization with EmbeddingConfig."""
        config = EmbeddingConfig(
            model_type="openai",
            model_name="text-embedding-3-small",
            batch_size=64
        )
        
        embedder = ConfigurableEmbeddings(config=config)
        
        self.assertEqual(embedder.model_type, "openai")
        self.assertEqual(embedder.model_name, "text-embedding-3-small")
        self.assertEqual(embedder.batch_size, 64)
    
    def test_initialization_with_none_config(self):
        """Test initialization with no config (defaults)."""
        embedder = ConfigurableEmbeddings()
        
        self.assertEqual(embedder.model_type, "sentence_transformer")
        self.assertIsNotNone(embedder.model_name)
    
    @patch('ai_vectorstore.components.embedding_model.configurable_embeddings.SentenceTransformer')
    def test_setup_sentence_transformer(self, mock_st):
        """Test setup with sentence transformer."""
        # Mock the model
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        config = EmbeddingConfig(model_type="sentence_transformer")
        embedder = ConfigurableEmbeddings(config=config)
        embedder.setup()
        
        self.assertTrue(embedder.initialized)
        self.assertEqual(embedder.embedding_dimension, 384)
        mock_st.assert_called_once()
    
    @patch('ai_vectorstore.components.embedding_model.configurable_embeddings.OpenAI')
    def test_setup_openai(self, mock_openai):
        """Test setup with OpenAI."""
        # Mock the client and embedding response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        os.environ["OPENAI_API_KEY"] = "test_key"
        
        config = EmbeddingConfig(model_type="openai")
        embedder = ConfigurableEmbeddings(config=config)
        embedder.setup()
        
        self.assertTrue(embedder.initialized)
        self.assertEqual(embedder.embedding_dimension, 1536)
        
        del os.environ["OPENAI_API_KEY"]
    
    def test_setup_openai_missing_key(self):
        """Test setup with OpenAI but missing API key."""
        config = EmbeddingConfig(model_type="openai", api_key=None)
        embedder = ConfigurableEmbeddings(config=config)
        
        # Make sure env var is not set
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        with self.assertRaises(ValueError) as cm:
            embedder.setup()
        
        self.assertIn("API key not found", str(cm.exception))
    
    def test_setup_unsupported_model_type(self):
        """Test setup with unsupported model type."""
        config = EmbeddingConfig(model_type="unsupported_type")
        embedder = ConfigurableEmbeddings(config=config)
        
        with self.assertRaises(ValueError) as cm:
            embedder.setup()
        
        self.assertIn("Unsupported model type", str(cm.exception))
    
    @patch('ai_vectorstore.components.embedding_model.configurable_embeddings.SentenceTransformer')
    def test_encode_texts(self, mock_st):
        """Test encoding texts."""
        # Mock model
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_st.return_value = mock_model
        
        config = EmbeddingConfig(model_type="sentence_transformer", normalize=False)
        embedder = ConfigurableEmbeddings(config=config)
        embedder.setup()
        
        texts = ["text1", "text2"]
        embeddings = embedder.run(texts)
        
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 3)
    
    @patch('ai_vectorstore.components.embedding_model.configurable_embeddings.SentenceTransformer')
    def test_encode_with_normalization(self, mock_st):
        """Test encoding with normalization."""
        import numpy as np
        
        # Mock model
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_model.encode.return_value = np.array([[1.0, 2.0, 2.0]])
        mock_st.return_value = mock_model
        
        config = EmbeddingConfig(model_type="sentence_transformer", normalize=True)
        embedder = ConfigurableEmbeddings(config=config)
        embedder.setup()
        
        embeddings = embedder.run(["text"])
        
        # Check normalization (L2 norm should be 1)
        import numpy as np
        norm = np.linalg.norm(embeddings[0])
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    @patch('ai_vectorstore.components.embedding_model.configurable_embeddings.SentenceTransformer')
    def test_encode_query(self, mock_st):
        """Test encoding single query."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_st.return_value = mock_model
        
        config = EmbeddingConfig(model_type="sentence_transformer", normalize=False)
        embedder = ConfigurableEmbeddings(config=config)
        embedder.setup()
        
        embedding = embedder.encode_query("query")
        
        self.assertEqual(len(embedding), 3)
        self.assertIsInstance(embedding, list)
    
    @patch('ai_vectorstore.components.embedding_model.configurable_embeddings.SentenceTransformer')
    def test_encode_documents(self, mock_st):
        """Test encoding multiple documents."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_st.return_value = mock_model
        
        config = EmbeddingConfig(model_type="sentence_transformer", normalize=False)
        embedder = ConfigurableEmbeddings(config=config)
        embedder.setup()
        
        embeddings = embedder.encode_documents(["doc1", "doc2"])
        
        self.assertEqual(len(embeddings), 2)
    
    @patch('ai_vectorstore.components.embedding_model.configurable_embeddings.SentenceTransformer')
    def test_embedding_dimension_property(self, mock_st):
        """Test embedding_dimension property."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        
        embedder = ConfigurableEmbeddings()
        
        # Should auto-setup if not initialized
        dim = embedder.embedding_dimension
        
        self.assertEqual(dim, 768)
        self.assertTrue(embedder.initialized)
    
    def test_encode_empty_list(self):
        """Test encoding empty list."""
        embedder = ConfigurableEmbeddings()
        result = embedder.run([])
        
        self.assertEqual(result, [])
    
    @patch('ai_vectorstore.components.embedding_model.configurable_embeddings.SentenceTransformer')
    def test_shutdown(self, mock_st):
        """Test shutdown."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        embedder = ConfigurableEmbeddings()
        embedder.setup()
        
        self.assertTrue(embedder.initialized)
        
        embedder.shutdown()
        
        self.assertFalse(embedder.initialized)
        self.assertIsNone(embedder._model)


class TestCreateEmbeddingModel(unittest.TestCase):
    """Test factory function."""
    
    def test_default_creation(self):
        """Test creating embedding model with defaults."""
        model = create_embedding_model()
        
        self.assertIsInstance(model, ConfigurableEmbeddings)
        self.assertEqual(model.model_type, "sentence_transformer")
    
    def test_custom_creation(self):
        """Test creating embedding model with custom parameters."""
        model = create_embedding_model(
            model_type="openai",
            model_name="text-embedding-3-small",
            batch_size=64
        )
        
        self.assertEqual(model.model_type, "openai")
        self.assertEqual(model.model_name, "text-embedding-3-small")
        self.assertEqual(model.batch_size, 64)
    
    def test_creation_with_kwargs(self):
        """Test creation with additional kwargs."""
        model = create_embedding_model(
            model_type="sentence_transformer",
            normalize=False,
            dimensions=512
        )
        
        self.assertFalse(model.embedding_config.normalize)
        self.assertEqual(model.embedding_config.dimensions, 512)


if __name__ == '__main__':
    unittest.main()


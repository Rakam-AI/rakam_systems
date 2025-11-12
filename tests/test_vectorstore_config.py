"""
Unit tests for Vector Store configuration system.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from rakam_systems.ai_vectorstore.config import (
    DatabaseConfig,
    EmbeddingConfig,
    IndexConfig,
    SearchConfig,
    VectorStoreConfig,
    load_config,
)


class TestEmbeddingConfig(unittest.TestCase):
    """Test EmbeddingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        
        self.assertEqual(config.model_type, "sentence_transformer")
        self.assertEqual(config.model_name, "Snowflake/snowflake-arctic-embed-m")
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.normalize)
        self.assertIsNone(config.dimensions)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EmbeddingConfig(
            model_type="openai",
            model_name="text-embedding-3-small",
            batch_size=64,
            normalize=False,
            dimensions=1536
        )
        
        self.assertEqual(config.model_type, "openai")
        self.assertEqual(config.model_name, "text-embedding-3-small")
        self.assertEqual(config.batch_size, 64)
        self.assertFalse(config.normalize)
        self.assertEqual(config.dimensions, 1536)
    
    def test_api_key_from_env(self):
        """Test API key loading from environment."""
        # Set environment variable
        os.environ["OPENAI_API_KEY"] = "test_key_123"
        
        config = EmbeddingConfig(model_type="openai")
        self.assertEqual(config.api_key, "test_key_123")
        
        # Cleanup
        del os.environ["OPENAI_API_KEY"]


class TestDatabaseConfig(unittest.TestCase):
    """Test DatabaseConfig class."""
    
    def test_default_config(self):
        """Test default database configuration."""
        config = DatabaseConfig()
        
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 5432)
        self.assertEqual(config.database, "vectorstore_db")
        self.assertEqual(config.user, "postgres")
        self.assertEqual(config.password, "postgres")
    
    def test_connection_string(self):
        """Test connection string generation."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            database="mydb",
            user="admin",
            password="secret"
        )
        
        expected = "postgresql://admin:secret@db.example.com:5433/mydb"
        self.assertEqual(config.to_connection_string(), expected)
    
    def test_env_override(self):
        """Test environment variable override."""
        os.environ["POSTGRES_HOST"] = "test.host"
        os.environ["POSTGRES_PORT"] = "5555"
        
        config = DatabaseConfig()
        
        self.assertEqual(config.host, "test.host")
        self.assertEqual(config.port, 5555)
        
        # Cleanup
        del os.environ["POSTGRES_HOST"]
        del os.environ["POSTGRES_PORT"]


class TestSearchConfig(unittest.TestCase):
    """Test SearchConfig class."""
    
    def test_default_config(self):
        """Test default search configuration."""
        config = SearchConfig()
        
        self.assertEqual(config.similarity_metric, "cosine")
        self.assertEqual(config.default_top_k, 5)
        self.assertTrue(config.enable_hybrid_search)
        self.assertEqual(config.hybrid_alpha, 0.7)
        self.assertTrue(config.rerank)
    
    def test_validation_success(self):
        """Test successful validation."""
        config = SearchConfig(
            similarity_metric="l2",
            default_top_k=10,
            hybrid_alpha=0.5
        )
        
        # Should not raise
        config.validate()
    
    def test_validation_invalid_metric(self):
        """Test validation with invalid metric."""
        config = SearchConfig(similarity_metric="invalid")
        
        with self.assertRaises(ValueError) as cm:
            config.validate()
        
        self.assertIn("Invalid similarity metric", str(cm.exception))
    
    def test_validation_invalid_alpha(self):
        """Test validation with invalid alpha."""
        config = SearchConfig(hybrid_alpha=1.5)
        
        with self.assertRaises(ValueError) as cm:
            config.validate()
        
        self.assertIn("hybrid_alpha must be between 0 and 1", str(cm.exception))
    
    def test_validation_invalid_top_k(self):
        """Test validation with invalid top_k."""
        config = SearchConfig(default_top_k=0)
        
        with self.assertRaises(ValueError) as cm:
            config.validate()
        
        self.assertIn("default_top_k must be >= 1", str(cm.exception))


class TestIndexConfig(unittest.TestCase):
    """Test IndexConfig class."""
    
    def test_default_config(self):
        """Test default index configuration."""
        config = IndexConfig()
        
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.chunk_overlap, 50)
        self.assertFalse(config.enable_parallel_processing)
        self.assertEqual(config.parallel_workers, 4)
        self.assertEqual(config.batch_insert_size, 100)


class TestVectorStoreConfig(unittest.TestCase):
    """Test VectorStoreConfig class."""
    
    def test_default_config(self):
        """Test default master configuration."""
        config = VectorStoreConfig()
        
        self.assertEqual(config.name, "pg_vector_store")
        self.assertIsInstance(config.embedding, EmbeddingConfig)
        self.assertIsInstance(config.database, DatabaseConfig)
        self.assertIsInstance(config.search, SearchConfig)
        self.assertIsInstance(config.index, IndexConfig)
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_size, 1000)
    
    def test_from_dict(self):
        """Test loading from dictionary."""
        config_dict = {
            "name": "test_store",
            "embedding": {
                "model_type": "openai",
                "model_name": "text-embedding-3-small",
                "batch_size": 64
            },
            "database": {
                "host": "db.test.com",
                "port": 5433
            },
            "search": {
                "similarity_metric": "l2",
                "default_top_k": 10
            },
            "enable_caching": False,
            "cache_size": 500
        }
        
        config = VectorStoreConfig.from_dict(config_dict)
        
        self.assertEqual(config.name, "test_store")
        self.assertEqual(config.embedding.model_type, "openai")
        self.assertEqual(config.database.host, "db.test.com")
        self.assertEqual(config.search.similarity_metric, "l2")
        self.assertFalse(config.enable_caching)
        self.assertEqual(config.cache_size, 500)
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        config = VectorStoreConfig(name="test_store")
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["name"], "test_store")
        self.assertIn("embedding", config_dict)
        self.assertIn("database", config_dict)
        self.assertIn("search", config_dict)
        self.assertIn("index", config_dict)
    
    def test_yaml_round_trip(self):
        """Test saving and loading YAML."""
        config = VectorStoreConfig(
            name="yaml_test",
            enable_caching=False,
            cache_size=2000
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            yaml_path = f.name
        
        try:
            loaded_config = VectorStoreConfig.from_yaml(yaml_path)
            
            self.assertEqual(loaded_config.name, "yaml_test")
            self.assertFalse(loaded_config.enable_caching)
            self.assertEqual(loaded_config.cache_size, 2000)
        finally:
            os.unlink(yaml_path)
    
    def test_json_round_trip(self):
        """Test saving and loading JSON."""
        config = VectorStoreConfig(
            name="json_test",
            enable_logging=False,
            log_level="DEBUG"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_json(f.name)
            json_path = f.name
        
        try:
            loaded_config = VectorStoreConfig.from_json(json_path)
            
            self.assertEqual(loaded_config.name, "json_test")
            self.assertFalse(loaded_config.enable_logging)
            self.assertEqual(loaded_config.log_level, "DEBUG")
        finally:
            os.unlink(json_path)
    
    def test_validation(self):
        """Test configuration validation."""
        config = VectorStoreConfig()
        
        # Should not raise
        config.validate()
    
    def test_validation_fails(self):
        """Test validation failure."""
        config = VectorStoreConfig()
        config.search.similarity_metric = "invalid"
        
        with self.assertRaises(ValueError):
            config.validate()


class TestLoadConfig(unittest.TestCase):
    """Test load_config function."""
    
    def test_load_none(self):
        """Test loading with None returns defaults."""
        config = load_config(None)
        
        self.assertIsInstance(config, VectorStoreConfig)
        self.assertEqual(config.name, "pg_vector_store")
    
    def test_load_dict(self):
        """Test loading from dictionary."""
        config_dict = {
            "name": "dict_test",
            "enable_caching": False
        }
        
        config = load_config(config_dict)
        
        self.assertEqual(config.name, "dict_test")
        self.assertFalse(config.enable_caching)
    
    def test_load_yaml_file(self):
        """Test loading from YAML file."""
        config_dict = {
            "name": "yaml_file_test",
            "embedding": {
                "model_type": "sentence_transformer",
                "batch_size": 16
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            yaml_path = f.name
        
        try:
            config = load_config(yaml_path)
            
            self.assertEqual(config.name, "yaml_file_test")
            self.assertEqual(config.embedding.batch_size, 16)
        finally:
            os.unlink(yaml_path)
    
    def test_load_json_file(self):
        """Test loading from JSON file."""
        config_dict = {
            "name": "json_file_test",
            "database": {
                "host": "json.test.com"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            json_path = f.name
        
        try:
            config = load_config(json_path)
            
            self.assertEqual(config.name, "json_file_test")
            self.assertEqual(config.database.host, "json.test.com")
        finally:
            os.unlink(json_path)
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with self.assertRaises(FileNotFoundError):
            load_config("/path/to/nonexistent/file.yaml")
    
    def test_auto_detect_yaml(self):
        """Test auto-detection of YAML files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({"name": "auto_yaml"}, f)
            yaml_path = f.name
        
        try:
            config = load_config(yaml_path, config_type='auto')
            self.assertEqual(config.name, "auto_yaml")
        finally:
            os.unlink(yaml_path)
    
    def test_auto_detect_json(self):
        """Test auto-detection of JSON files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"name": "auto_json"}, f)
            json_path = f.name
        
        try:
            config = load_config(json_path, config_type='auto')
            self.assertEqual(config.name, "auto_json")
        finally:
            os.unlink(json_path)


if __name__ == '__main__':
    unittest.main()


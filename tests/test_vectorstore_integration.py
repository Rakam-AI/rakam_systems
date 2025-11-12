"""
Integration tests for Vector Store full lifecycle.

These tests require:
- PostgreSQL with pgvector extension
- Django settings configured
- Database connection available

Run with: python -m pytest tests/test_vectorstore_integration.py
"""

import os
import unittest
from typing import List

# Configure Django before imports
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "examples.ai_vectorstore_examples.django_settings"
)

try:
    import django
    django.setup()
    DJANGO_AVAILABLE = True
except Exception:
    DJANGO_AVAILABLE = False


@unittest.skipUnless(DJANGO_AVAILABLE, "Django not configured")
class TestVectorStoreIntegration(unittest.TestCase):
    """Integration tests for complete vector store lifecycle."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test class."""
        if not DJANGO_AVAILABLE:
            return
        
        from rakam_systems.ai_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore
        from rakam_systems.ai_vectorstore.components.loader.adaptive_loader import AdaptiveLoader
        from rakam_systems.ai_vectorstore.config import VectorStoreConfig, EmbeddingConfig
        
        cls.ConfigurablePgVectorStore = ConfigurablePgVectorStore
        cls.AdaptiveLoader = AdaptiveLoader
        cls.VectorStoreConfig = VectorStoreConfig
        cls.EmbeddingConfig = EmbeddingConfig
    
    def setUp(self):
        """Setup test fixtures."""
        if not DJANGO_AVAILABLE:
            self.skipTest("Django not available")
        
        from rakam_systems.ai_vectorstore.core import Node, NodeMetadata
        
        # Create test configuration
        config = self.VectorStoreConfig(
            name="test_vector_store",
            embedding=self.EmbeddingConfig(
                model_type="sentence_transformer",
                model_name="all-MiniLM-L6-v2",  # Small fast model for testing
                batch_size=8
            )
        )
        
        self.store = self.ConfigurablePgVectorStore(config=config)
        self.store.setup()
        
        self.loader = self.AdaptiveLoader()
        
        self.test_collection = "test_integration_collection"
        self.Node = Node
        self.NodeMetadata = NodeMetadata
        
        # Clean up any existing test collection
        try:
            self.store.delete_collection(self.test_collection)
        except Exception:
            pass
    
    def tearDown(self):
        """Cleanup after tests."""
        if not DJANGO_AVAILABLE:
            return
        
        # Clean up test collection
        try:
            self.store.delete_collection(self.test_collection)
        except Exception:
            pass
        
        if hasattr(self, 'store'):
            self.store.shutdown()
    
    def test_full_lifecycle(self):
        """Test complete vector store lifecycle."""
        # 1. Create sample nodes
        documents = [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neural networks.",
            "PostgreSQL is a powerful relational database.",
            "Vector databases enable semantic search."
        ]
        
        nodes = []
        for idx, doc in enumerate(documents):
            metadata = self.NodeMetadata(
                source_file_uuid="test_docs",
                position=idx,
                custom={"category": "tech", "index": idx}
            )
            node = self.Node(content=doc, metadata=metadata)
            nodes.append(node)
        
        # 2. Create collection from nodes
        self.store.create_collection_from_nodes(self.test_collection, nodes)
        
        # 3. Verify collection was created
        collections = self.store.list_collections()
        self.assertIn(self.test_collection, collections)
        
        # 4. Get collection info
        info = self.store.get_collection_info(self.test_collection)
        self.assertEqual(info["node_count"], len(documents))
        self.assertGreater(info["embedding_dim"], 0)
        
        # 5. Search for similar content
        query = "What is artificial intelligence?"
        results, result_nodes = self.store.search(
            collection_name=self.test_collection,
            query=query,
            number=3
        )
        
        # Verify search results
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)
        self.assertEqual(len(results), len(result_nodes))
        
        # 6. Add more nodes
        new_documents = [
            "Deep learning uses multiple layers in neural networks.",
            "Data science combines statistics and programming."
        ]
        
        new_nodes = []
        for idx, doc in enumerate(new_documents):
            metadata = self.NodeMetadata(
                source_file_uuid="additional_docs",
                position=idx,
                custom={"category": "tech", "batch": "2"}
            )
            node = self.Node(content=doc, metadata=metadata)
            new_nodes.append(node)
        
        self.store.add_nodes(self.test_collection, new_nodes)
        
        # 7. Verify node count increased
        info = self.store.get_collection_info(self.test_collection)
        self.assertEqual(info["node_count"], len(documents) + len(new_documents))
        
        # 8. Update a vector
        first_node_id = result_nodes[0].metadata.node_id
        new_content = "Python is the most popular programming language for data science."
        
        self.store.update_vector(
            collection_name=self.test_collection,
            node_id=first_node_id,
            new_content=new_content,
            new_metadata={"updated": True}
        )
        
        # 9. Search with metadata filters
        filtered_results, filtered_nodes = self.store.search(
            collection_name=self.test_collection,
            query="programming",
            number=5,
            meta_data_filters={"category": "tech"}
        )
        
        self.assertGreater(len(filtered_results), 0)
        
        # 10. Delete some nodes
        node_ids_to_delete = [result_nodes[0].metadata.node_id]
        self.store.delete_nodes(self.test_collection, node_ids_to_delete)
        
        # 11. Verify deletion
        info = self.store.get_collection_info(self.test_collection)
        self.assertEqual(info["node_count"], len(documents) + len(new_documents) - 1)
        
        # 12. Delete collection
        self.store.delete_collection(self.test_collection)
        
        # 13. Verify collection is gone
        collections = self.store.list_collections()
        self.assertNotIn(self.test_collection, collections)
    
    def test_adaptive_loader_integration(self):
        """Test integration with adaptive loader."""
        import tempfile
        
        # Create test text file
        test_content = """
# Vector Databases

Vector databases are specialized databases designed for storing and querying high-dimensional vectors.

## Features
- Fast similarity search
- Scalable architecture
- Support for various distance metrics

## Use Cases
Vector databases are commonly used in:
- Semantic search
- Recommendation systems
- Image similarity
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            # Load file with adaptive loader
            nodes = self.loader.load_as_nodes(
                temp_file,
                source_id="test_markdown",
                custom_metadata={"type": "documentation"}
            )
            
            self.assertGreater(len(nodes), 0)
            
            # Create collection from loaded nodes
            self.store.create_collection_from_nodes(self.test_collection, nodes)
            
            # Search in the collection
            results, result_nodes = self.store.search(
                collection_name=self.test_collection,
                query="What are vector databases used for?",
                number=3
            )
            
            self.assertGreater(len(results), 0)
            
            # Verify metadata was preserved
            for node in result_nodes:
                self.assertEqual(node.metadata.custom.get("type"), "documentation")
        
        finally:
            os.unlink(temp_file)
    
    def test_configuration_persistence(self):
        """Test that configuration is properly applied."""
        # Verify configuration is active
        self.assertEqual(self.store.vs_config.name, "test_vector_store")
        self.assertEqual(self.store.vs_config.embedding.model_name, "all-MiniLM-L6-v2")
        
        # Verify embedding dimension matches config
        self.assertIsNotNone(self.store.embedding_dim)
        self.assertGreater(self.store.embedding_dim, 0)
    
    def test_hybrid_search(self):
        """Test hybrid search functionality."""
        # Create nodes with varied content
        documents = [
            "Python programming language for data science",
            "Machine learning algorithms and models",
            "SQL databases and queries",
            "Python web development with Django",
            "Data analysis using Python pandas"
        ]
        
        nodes = []
        for idx, doc in enumerate(documents):
            metadata = self.NodeMetadata(
                source_file_uuid="hybrid_test",
                position=idx,
                custom={}
            )
            node = self.Node(content=doc, metadata=metadata)
            nodes.append(node)
        
        self.store.create_collection_from_nodes(self.test_collection, nodes)
        
        # Search with hybrid search enabled
        query = "Python data"
        results_hybrid, nodes_hybrid = self.store.search(
            collection_name=self.test_collection,
            query=query,
            number=3,
            hybrid_search=True
        )
        
        # Search without hybrid search
        results_vector, nodes_vector = self.store.search(
            collection_name=self.test_collection,
            query=query,
            number=3,
            hybrid_search=False
        )
        
        # Both should return results
        self.assertGreater(len(results_hybrid), 0)
        self.assertGreater(len(results_vector), 0)
    
    def test_different_similarity_metrics(self):
        """Test different similarity metrics."""
        documents = [
            "Test document one",
            "Test document two",
            "Test document three"
        ]
        
        nodes = []
        for idx, doc in enumerate(documents):
            metadata = self.NodeMetadata(
                source_file_uuid="metric_test",
                position=idx,
                custom={}
            )
            node = self.Node(content=doc, metadata=metadata)
            nodes.append(node)
        
        self.store.create_collection_from_nodes(self.test_collection, nodes)
        
        query = "test document"
        
        # Test cosine similarity
        results_cosine, _ = self.store.search(
            collection_name=self.test_collection,
            query=query,
            distance_type="cosine",
            number=2
        )
        
        # Test L2 distance
        results_l2, _ = self.store.search(
            collection_name=self.test_collection,
            query=query,
            distance_type="l2",
            number=2
        )
        
        # Both should return results
        self.assertEqual(len(results_cosine), 2)
        self.assertEqual(len(results_l2), 2)


@unittest.skipUnless(DJANGO_AVAILABLE, "Django not configured")
class TestVectorStoreInterface(unittest.TestCase):
    """Test VectorStore interface compatibility."""
    
    def setUp(self):
        """Setup test fixtures."""
        if not DJANGO_AVAILABLE:
            self.skipTest("Django not available")
        
        from rakam_systems.ai_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore
        from rakam_systems.ai_vectorstore.config import VectorStoreConfig, EmbeddingConfig
        
        config = VectorStoreConfig(
            embedding=EmbeddingConfig(
                model_type="sentence_transformer",
                model_name="all-MiniLM-L6-v2"
            )
        )
        
        self.store = ConfigurablePgVectorStore(config=config)
        self.store.setup()
        
        self.test_collection = "test_interface_collection"
        
        # Clean up
        try:
            self.store.delete_collection(self.test_collection)
        except Exception:
            pass
    
    def tearDown(self):
        """Cleanup."""
        if not DJANGO_AVAILABLE:
            return
        
        try:
            self.store.delete_collection(self.test_collection)
        except Exception:
            pass
        
        if hasattr(self, 'store'):
            self.store.shutdown()
    
    def test_add_and_query_interface(self):
        """Test VectorStore add() and query() interface methods."""
        # Generate sample embeddings
        vectors = [
            [0.1, 0.2, 0.3] * 128,  # 384-dim vector
            [0.4, 0.5, 0.6] * 128,
            [0.7, 0.8, 0.9] * 128
        ]
        
        metadatas = [
            {"content": "Document 1", "collection_name": self.test_collection, "category": "A"},
            {"content": "Document 2", "collection_name": self.test_collection, "category": "B"},
            {"content": "Document 3", "collection_name": self.test_collection, "category": "A"}
        ]
        
        # Add vectors
        node_ids = self.store.add(vectors, metadatas)
        
        self.assertEqual(len(node_ids), 3)
        
        # Query vectors
        query_vector = [0.1, 0.2, 0.3] * 128
        results = self.store.query(
            query_vector,
            top_k=2,
            collection_name=self.test_collection
        )
        
        self.assertLessEqual(len(results), 2)
        
        # Verify result structure
        for result in results:
            self.assertIn("node_id", result)
            self.assertIn("content", result)
            self.assertIn("metadata", result)
            self.assertIn("distance", result)


if __name__ == '__main__':
    unittest.main()


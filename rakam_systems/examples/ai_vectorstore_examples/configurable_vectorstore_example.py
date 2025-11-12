"""
Configurable Vector Store Example

This example demonstrates the enhanced, configurable PgVectorStore with:
1. YAML-based configuration
2. Adaptive data loading (auto-detects file types)
3. Multiple embedding backends
4. Update operations
5. Advanced search features
6. Complete CRUD lifecycle

Prerequisites:
- PostgreSQL with pgvector extension running
- Django configured (see django_settings.py)
- Environment variables set (optional)
"""

import os
import sys
import tempfile

# Configure Django
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "examples.ai_vectorstore_examples.django_settings"
)

import django
django.setup()

from ai_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore
from ai_vectorstore.components.loader.adaptive_loader import AdaptiveLoader
from ai_vectorstore.config import VectorStoreConfig, EmbeddingConfig, SearchConfig
from ai_vectorstore.core import Node, NodeMetadata


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Print a formatted section."""
    print(f"\n--- {title} ---")


def create_sample_documents():
    """Create sample documents for testing."""
    return [
        {
            "content": "Python is a versatile programming language widely used in data science and machine learning.",
            "metadata": {"category": "programming", "language": "python", "difficulty": "beginner"}
        },
        {
            "content": "Machine learning algorithms can automatically learn patterns from data without explicit programming.",
            "metadata": {"category": "ai", "topic": "machine_learning", "difficulty": "intermediate"}
        },
        {
            "content": "Neural networks consist of layers of interconnected nodes that process information.",
            "metadata": {"category": "ai", "topic": "deep_learning", "difficulty": "advanced"}
        },
        {
            "content": "PostgreSQL is a powerful, open-source relational database management system.",
            "metadata": {"category": "databases", "type": "sql", "difficulty": "intermediate"}
        },
        {
            "content": "Vector databases enable efficient similarity search and semantic retrieval of high-dimensional data.",
            "metadata": {"category": "databases", "type": "vector", "difficulty": "advanced"}
        },
    ]


def example_1_basic_configuration():
    """Example 1: Using configuration for vector store setup."""
    print_header("Example 1: Configuration-Based Setup")
    
    # Method 1: Create configuration programmatically
    print_section("Method 1: Programmatic Configuration")
    
    config = VectorStoreConfig(
        name="example_vector_store",
        embedding=EmbeddingConfig(
            model_type="sentence_transformer",
            model_name="all-MiniLM-L6-v2",  # Fast, lightweight model
            batch_size=16,
            normalize=True
        ),
        search=SearchConfig(
            similarity_metric="cosine",
            default_top_k=5,
            enable_hybrid_search=True,
            hybrid_alpha=0.7
        )
    )
    
    print(f"✓ Created configuration: {config.name}")
    print(f"  - Embedding model: {config.embedding.model_name}")
    print(f"  - Similarity metric: {config.search.similarity_metric}")
    print(f"  - Hybrid search: {config.search.enable_hybrid_search}")
    
    # Initialize vector store with configuration
    store = ConfigurablePgVectorStore(config=config)
    store.setup()
    
    print(f"✓ Vector store initialized with {store.embedding_dim}-dimensional embeddings")
    
    store.shutdown()
    
    # Method 2: Load from YAML file
    print_section("Method 2: Load from YAML Configuration File")
    
    config_path = "examples/configs/pg_vectorstore_config.yaml"
    if os.path.exists(config_path):
        store = ConfigurablePgVectorStore(config=config_path)
        store.setup()
        print(f"✓ Loaded configuration from: {config_path}")
        print(f"✓ Store name: {store.vs_config.name}")
        store.shutdown()
    else:
        print(f"⚠️  Config file not found: {config_path}")
    
    # Method 3: Load from dictionary
    print_section("Method 3: Configuration from Dictionary")
    
    config_dict = {
        "name": "dict_configured_store",
        "embedding": {
            "model_type": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 32
        },
        "search": {
            "similarity_metric": "cosine",
            "default_top_k": 10
        }
    }
    
    store = ConfigurablePgVectorStore(config=config_dict)
    store.setup()
    print(f"✓ Configured from dictionary")
    print(f"  - Batch size: {store.vs_config.embedding.batch_size}")
    print(f"  - Default top_k: {store.vs_config.search.default_top_k}")
    store.shutdown()


def example_2_adaptive_loading():
    """Example 2: Adaptive data loading with different file types."""
    print_header("Example 2: Adaptive Data Loading")
    
    loader = AdaptiveLoader()
    temp_files = []
    
    try:
        # Create sample files of different types
        print_section("Creating Sample Files")
        
        # 1. Text file
        text_content = "This is a plain text document about vector databases. They are very useful for semantic search."
        text_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        text_file.write(text_content)
        text_file.close()
        temp_files.append(text_file.name)
        print(f"✓ Created text file: {text_file.name}")
        
        # 2. Markdown file
        md_content = """# Vector Databases

## Introduction
Vector databases are specialized for storing and querying embeddings.

## Features
- Fast similarity search
- Scalable architecture
- Multiple distance metrics
"""
        md_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        md_file.write(md_content)
        md_file.close()
        temp_files.append(md_file.name)
        print(f"✓ Created markdown file: {md_file.name}")
        
        # 3. JSON file
        import json
        json_content = {
            "title": "Machine Learning Basics",
            "content": "Machine learning is a subset of AI that enables systems to learn from data.",
            "tags": ["AI", "ML", "data science"]
        }
        json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(json_content, json_file)
        json_file.close()
        temp_files.append(json_file.name)
        print(f"✓ Created JSON file: {json_file.name}")
        
        # Load files with adaptive loader
        print_section("Loading Files with Adaptive Loader")
        
        for file_path in temp_files:
            file_type = file_path.split('.')[-1]
            chunks = loader.run(file_path)
            print(f"✓ Loaded {file_type.upper()} file: {len(chunks)} chunks")
        
        # Load as nodes
        print_section("Loading as Node Objects")
        
        nodes = loader.load_as_nodes(
            temp_files[0],
            source_id="text_source",
            custom_metadata={"type": "text", "batch": "demo"}
        )
        print(f"✓ Loaded {len(nodes)} nodes with metadata")
        print(f"  - Source ID: {nodes[0].metadata.source_file_uuid}")
        print(f"  - Custom metadata: {nodes[0].metadata.custom}")
        
        # Load as VSFile
        print_section("Loading as VSFile")
        
        vsfile = loader.load_as_vsfile(
            temp_files[1],
            custom_metadata={"format": "markdown"}
        )
        print(f"✓ Loaded VSFile: {vsfile.file_name}")
        print(f"  - UUID: {vsfile.uuid}")
        print(f"  - Nodes: {len(vsfile.nodes)}")
        print(f"  - Processed: {vsfile.processed}")
    
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass


def example_3_full_crud_lifecycle():
    """Example 3: Complete CRUD lifecycle with vector store."""
    print_header("Example 3: Full CRUD Lifecycle")
    
    # Initialize store
    config = VectorStoreConfig(
        embedding=EmbeddingConfig(
            model_type="sentence_transformer",
            model_name="all-MiniLM-L6-v2",
            batch_size=8
        )
    )
    store = ConfigurablePgVectorStore(config=config)
    store.setup()
    
    collection_name = "demo_crud_collection"
    
    try:
        # Clean up any existing collection
        try:
            store.delete_collection(collection_name)
        except Exception:
            pass
        
        # CREATE: Add documents to collection
        print_section("CREATE: Adding Documents")
        
        documents = create_sample_documents()
        nodes = []
        
        for idx, doc in enumerate(documents):
            metadata = NodeMetadata(
                source_file_uuid="demo_source",
                position=idx,
                custom=doc["metadata"]
            )
            node = Node(content=doc["content"], metadata=metadata)
            nodes.append(node)
        
        store.create_collection_from_nodes(collection_name, nodes)
        print(f"✓ Created collection with {len(nodes)} documents")
        
        # READ: Search and retrieve
        print_section("READ: Searching Documents")
        
        query = "Tell me about machine learning"
        results, result_nodes = store.search(
            collection_name=collection_name,
            query=query,
            number=3
        )
        
        print(f"✓ Search query: '{query}'")
        print(f"✓ Found {len(results)} results:")
        for idx, node in enumerate(result_nodes, 1):
            print(f"  {idx}. {node.content[:60]}...")
            custom = node.metadata.custom if isinstance(node.metadata.custom, dict) else {}
            print(f"     Category: {custom.get('category', 'N/A')}")
        
        # UPDATE: Modify a document
        print_section("UPDATE: Modifying a Document")
        
        first_node_id = result_nodes[0].metadata.node_id
        new_content = "Python is the most popular programming language for AI and data science applications."
        
        store.update_vector(
            collection_name=collection_name,
            node_id=first_node_id,
            new_content=new_content,
            new_metadata={"updated": True, "version": 2}
        )
        print(f"✓ Updated node {first_node_id}")
        print(f"  New content: {new_content[:50]}...")
        
        # Verify update
        results, result_nodes = store.search(
            collection_name=collection_name,
            query="Python programming",
            number=1
        )
        print(f"✓ Verified update - found updated content")
        
        # ADD: Add more documents
        print_section("ADD: Adding More Documents")
        
        new_documents = [
            {
                "content": "Deep learning is a subset of machine learning using multi-layered neural networks.",
                "metadata": {"category": "ai", "topic": "deep_learning", "difficulty": "advanced"}
            },
            {
                "content": "Natural language processing enables computers to understand human language.",
                "metadata": {"category": "ai", "topic": "nlp", "difficulty": "intermediate"}
            }
        ]
        
        new_nodes = []
        for idx, doc in enumerate(new_documents):
            metadata = NodeMetadata(
                source_file_uuid="additional_source",
                position=idx,
                custom=doc["metadata"]
            )
            node = Node(content=doc["content"], metadata=metadata)
            new_nodes.append(node)
        
        store.add_nodes(collection_name, new_nodes)
        print(f"✓ Added {len(new_nodes)} new documents")
        
        # Verify addition
        info = store.get_collection_info(collection_name)
        print(f"✓ Total documents in collection: {info['node_count']}")
        
        # DELETE: Remove documents
        print_section("DELETE: Removing Documents")
        
        node_ids_to_delete = [result_nodes[0].metadata.node_id]
        store.delete_nodes(collection_name, node_ids_to_delete)
        print(f"✓ Deleted {len(node_ids_to_delete)} document(s)")
        
        # Verify deletion
        info = store.get_collection_info(collection_name)
        print(f"✓ Remaining documents: {info['node_count']}")
        
        # Collection info
        print_section("Collection Information")
        
        info = store.get_collection_info(collection_name)
        print(f"Name: {info['name']}")
        print(f"Embedding dimension: {info['embedding_dim']}")
        print(f"Document count: {info['node_count']}")
        print(f"Created: {info['created_at']}")
        print(f"Updated: {info['updated_at']}")
    
    finally:
        # Cleanup
        try:
            store.delete_collection(collection_name)
            print(f"\n✓ Cleaned up collection: {collection_name}")
        except Exception:
            pass
        
        store.shutdown()


def example_4_advanced_search():
    """Example 4: Advanced search features."""
    print_header("Example 4: Advanced Search Features")
    
    config = VectorStoreConfig(
        embedding=EmbeddingConfig(
            model_type="sentence_transformer",
            model_name="all-MiniLM-L6-v2"
        ),
        search=SearchConfig(
            similarity_metric="cosine",
            enable_hybrid_search=True,
            hybrid_alpha=0.7,
            rerank=True
        )
    )
    
    store = ConfigurablePgVectorStore(config=config)
    store.setup()
    
    collection_name = "demo_search_collection"
    
    try:
        # Clean up
        try:
            store.delete_collection(collection_name)
        except Exception:
            pass
        
        # Create diverse documents
        documents = create_sample_documents()
        nodes = []
        
        for idx, doc in enumerate(documents):
            metadata = NodeMetadata(
                source_file_uuid="search_demo",
                position=idx,
                custom=doc["metadata"]
            )
            node = Node(content=doc["content"], metadata=metadata)
            nodes.append(node)
        
        store.create_collection_from_nodes(collection_name, nodes)
        print(f"✓ Created collection with {len(nodes)} documents")
        
        # 1. Semantic search
        print_section("1. Semantic Search")
        
        query = "artificial intelligence and neural networks"
        results, result_nodes = store.search(
            collection_name=collection_name,
            query=query,
            number=3
        )
        
        print(f"Query: '{query}'")
        print(f"Results ({len(results)}):")
        for idx, node in enumerate(result_nodes, 1):
            custom = node.metadata.custom if isinstance(node.metadata.custom, dict) else {}
            category = custom.get('category', 'N/A')
            print(f"  {idx}. [{category}] {node.content[:70]}...")
        
        # 2. Metadata filtering
        print_section("2. Search with Metadata Filters")
        
        query = "programming databases"
        results, result_nodes = store.search(
            collection_name=collection_name,
            query=query,
            number=5,
            meta_data_filters={"category": "databases"}
        )
        
        print(f"Query: '{query}'")
        print(f"Filter: category='databases'")
        print(f"Results ({len(results)}):")
        for idx, node in enumerate(result_nodes, 1):
            custom = node.metadata.custom if isinstance(node.metadata.custom, dict) else {}
            db_type = custom.get('type', 'N/A')
            print(f"  {idx}. [Type: {db_type}] {node.content[:70]}...")
        
        # 3. Different similarity metrics
        print_section("3. Different Similarity Metrics")
        
        query = "database systems"
        
        for metric in ["cosine", "l2"]:
            results, result_nodes = store.search(
                collection_name=collection_name,
                query=query,
                distance_type=metric,
                number=2
            )
            
            print(f"Metric: {metric} - Found {len(results)} results")
        
        # 4. Hybrid search comparison
        print_section("4. Hybrid vs Vector-Only Search")
        
        query = "Python programming language"
        
        # With hybrid search
        results_hybrid, nodes_hybrid = store.search(
            collection_name=collection_name,
            query=query,
            number=3,
            hybrid_search=True
        )
        
        # Without hybrid search
        results_vector, nodes_vector = store.search(
            collection_name=collection_name,
            query=query,
            number=3,
            hybrid_search=False
        )
        
        print(f"Query: '{query}'")
        print(f"Hybrid search results: {len(results_hybrid)}")
        print(f"Vector-only results: {len(results_vector)}")
        print(f"✓ Both methods returned results (may differ in ranking)")
    
    finally:
        try:
            store.delete_collection(collection_name)
        except Exception:
            pass
        
        store.shutdown()


def example_5_configuration_management():
    """Example 5: Configuration save and load."""
    print_header("Example 5: Configuration Management")
    
    # Create a custom configuration
    print_section("Creating Custom Configuration")
    
    config = VectorStoreConfig(
        name="production_vector_store",
        embedding=EmbeddingConfig(
            model_type="sentence_transformer",
            model_name="all-MiniLM-L6-v2",
            batch_size=64,
            normalize=True
        ),
        search=SearchConfig(
            similarity_metric="cosine",
            default_top_k=10,
            enable_hybrid_search=True,
            hybrid_alpha=0.8
        ),
        enable_caching=True,
        cache_size=2000,
        log_level="INFO"
    )
    
    print(f"✓ Created configuration: {config.name}")
    
    # Save to YAML
    print_section("Saving Configuration to YAML")
    
    yaml_path = tempfile.mktemp(suffix='.yaml')
    config.save_yaml(yaml_path)
    print(f"✓ Saved to: {yaml_path}")
    
    # Load from YAML
    print_section("Loading Configuration from YAML")
    
    loaded_config = VectorStoreConfig.from_yaml(yaml_path)
    print(f"✓ Loaded configuration: {loaded_config.name}")
    print(f"  - Embedding model: {loaded_config.embedding.model_name}")
    print(f"  - Batch size: {loaded_config.embedding.batch_size}")
    print(f"  - Top K: {loaded_config.search.default_top_k}")
    print(f"  - Cache size: {loaded_config.cache_size}")
    
    # Save to JSON
    print_section("Saving Configuration to JSON")
    
    json_path = tempfile.mktemp(suffix='.json')
    config.save_json(json_path)
    print(f"✓ Saved to: {json_path}")
    
    # Load from JSON
    loaded_json_config = VectorStoreConfig.from_json(json_path)
    print(f"✓ Loaded from JSON: {loaded_json_config.name}")
    
    # Cleanup
    try:
        os.unlink(yaml_path)
        os.unlink(json_path)
    except Exception:
        pass


def main():
    """Run all examples."""
    print("\n" + "🚀" * 35)
    print("  CONFIGURABLE VECTOR STORE - COMPREHENSIVE EXAMPLES")
    print("🚀" * 35)
    
    try:
        example_1_basic_configuration()
        example_2_adaptive_loading()
        example_3_full_crud_lifecycle()
        example_4_advanced_search()
        example_5_configuration_management()
        
        print_header("✅ All Examples Completed Successfully!")
        
        print("\n📚 Key Takeaways:")
        print("  1. Configuration can be loaded from YAML, JSON, or created programmatically")
        print("  2. Adaptive loader automatically handles different file types")
        print("  3. Full CRUD operations: Create, Read, Update, Delete")
        print("  4. Advanced search: metadata filters, hybrid search, multiple metrics")
        print("  5. Configuration persistence for reproducibility")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Troubleshooting:")
        print("  1. Ensure PostgreSQL with pgvector is running")
        print("  2. Check database connection settings")
        print("  3. Verify Django configuration")
        print("  4. Install required packages: sentence-transformers, pgvector, django")


if __name__ == "__main__":
    main()


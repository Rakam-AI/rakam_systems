"""
PostgreSQL Vector Store Example

This example demonstrates:
1. Using PgVectorStore with PostgreSQL + pgvector extension
2. Database-backed vector storage for production use
3. Hybrid search combining vector and full-text search
4. Re-ranking results for better relevance
5. CRUD operations with database persistence

Note: Requires PostgreSQL with pgvector extension and Django setup.
"""

import os
import sys

# Configure Django before importing any Django-dependent modules
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "examples.ai_vectorstore_examples.django_settings"
)

import django
django.setup()

from ai_vectorstore.core import Node, NodeMetadata


def create_sample_documents():
    """Create sample documents for demonstration."""
    documents = [
        {
            "content": "Python is a versatile programming language popular for data science.",
            "metadata": {"category": "programming", "language": "python", "difficulty": "beginner"}
        },
        {
            "content": "Machine learning algorithms can identify patterns in large datasets.",
            "metadata": {"category": "ai", "topic": "machine_learning", "difficulty": "intermediate"}
        },
        {
            "content": "Neural networks consist of interconnected layers of artificial neurons.",
            "metadata": {"category": "ai", "topic": "deep_learning", "difficulty": "advanced"}
        },
        {
            "content": "PostgreSQL is a powerful open-source relational database system.",
            "metadata": {"category": "databases", "type": "sql", "difficulty": "intermediate"}
        },
        {
            "content": "Vector databases enable efficient similarity search for embeddings.",
            "metadata": {"category": "databases", "type": "vector", "difficulty": "advanced"}
        },
        {
            "content": "Django is a high-level Python web framework for rapid development.",
            "metadata": {"category": "programming", "language": "python", "difficulty": "intermediate"}
        },
        {
            "content": "Natural language processing helps computers understand human language.",
            "metadata": {"category": "ai", "topic": "nlp", "difficulty": "intermediate"}
        },
        {
            "content": "Docker containers provide isolated environments for applications.",
            "metadata": {"category": "devops", "topic": "containerization", "difficulty": "intermediate"}
        },
    ]
    
    nodes = []
    for idx, doc in enumerate(documents):
        metadata = NodeMetadata(
            source_file_uuid="demo_docs",
            position=idx,
            custom=doc["metadata"]
        )
        node = Node(content=doc["content"], metadata=metadata)
        nodes.append(node)
    
    return nodes


def main():
    """
    Main example function demonstrating PgVectorStore usage.
    
    NOTE: This example requires:
    1. PostgreSQL installed with pgvector extension
    2. Environment variables configured (see below)
    3. Database tables created
    
    Environment Variables:
    - POSTGRES_DB: Database name (default: vectorstore_db)
    - POSTGRES_USER: Database user (default: postgres)
    - POSTGRES_PASSWORD: Database password (default: postgres)
    - POSTGRES_HOST: Database host (default: localhost)
    - POSTGRES_PORT: Database port (default: 5432)
    """
    
    print("=" * 60)
    print("PostgreSQL Vector Store Example")
    print("=" * 60)
    
    print("\n⚠️  Prerequisites Check:")
    print("   This example requires:")
    print("   1. PostgreSQL with pgvector extension installed")
    print("   2. PostgreSQL server running")
    print("   3. Database created and accessible")
    print("   4. Environment variables configured (optional)")
    
    # Check database connection
    try:
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"\n✓ Connected to PostgreSQL: {version[:50]}...")
    except Exception as e:
        print(f"\n❌ Cannot connect to PostgreSQL: {e}")
        print("\nPlease check:")
        print("1. PostgreSQL is running")
        print("2. Database exists and is accessible")
        print("3. Credentials are correct (set via environment variables)")
        print("\nEnvironment variables:")
        print(f"  POSTGRES_DB={os.getenv('POSTGRES_DB', 'vectorstore_db')}")
        print(f"  POSTGRES_USER={os.getenv('POSTGRES_USER', 'postgres')}")
        print(f"  POSTGRES_HOST={os.getenv('POSTGRES_HOST', 'localhost')}")
        print(f"  POSTGRES_PORT={os.getenv('POSTGRES_PORT', '5432')}")
        return
    
    # Import PgVectorStore
    try:
        from ai_vectorstore.components.vectorstore.pg_vector_store import PgVectorStore
    except ImportError as e:
        print(f"\n❌ Cannot import PgVectorStore: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install django psycopg2-binary pgvector sentence-transformers")
        return
    
    try:
        # Initialize PostgreSQL vector store
        print("\n1. Initializing PostgreSQL vector store...")
        # Use sentence-transformers/all-MiniLM-L6-v2 which produces 384 dimensions
        # matching the database schema in pg_models.py
        store = PgVectorStore(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            use_embedding_api=False
        )
        print("   ✓ PgVectorStore initialized")
        print(f"   ✓ Embedding dimension: {store.embedding_dim}")
        
        # Create sample documents
        print("\n2. Creating sample documents...")
        nodes = create_sample_documents()
        print(f"   ✓ Created {len(nodes)} sample nodes")
        
        # Create or get collection
        print("\n3. Creating collection...")
        collection_name = "demo_collection"
        collection = store.get_or_create_collection(collection_name)
        print(f"   ✓ Collection '{collection_name}' ready")
        
        # Add documents to collection
        print("\n4. Adding documents to collection...")
        store.create_collection_from_nodes(collection_name, nodes)
        print(f"   ✓ Added {len(nodes)} documents")
        
        # Get collection info
        print("\n5. Collection information...")
        info = store.get_collection_info(collection_name)
        print(f"   Name: {info['name']}")
        print(f"   Embedding dimension: {info['embedding_dim']}")
        print(f"   Node count: {info['node_count']}")
        print(f"   Created at: {info['created_at']}")
        
        # Perform semantic search
        print("\n6. Performing semantic search...")
        queries = [
            "What is artificial intelligence?",
            "Tell me about Python programming",
            "How do databases work?"
        ]
        
        for query in queries:
            print(f"\n   Query: '{query}'")
            results, result_nodes = store.search(
                collection_name=collection_name,
                query=query,
                distance_type="cosine",
                number=3,
                hybrid_search=True  # Enable hybrid search
            )
            
            print(f"   Top {len(results)} results (with hybrid search):")
            for idx, (node_id, (metadata, content, distance)) in enumerate(results.items(), 1):
                category = metadata['custom'].get('category', 'N/A')
                print(f"      {idx}. [{category}] {content[:70]}...")
        
        # Search with metadata filters
        print("\n7. Search with metadata filters...")
        query = "Tell me about programming"
        
        # Get all node IDs first
        all_results, all_nodes = store.search(
            collection_name=collection_name,
            query="",
            number=100
        )
        
        # Filter by category
        programming_filters = {
            "category": "programming"
        }
        
        print(f"   Query: '{query}'")
        print(f"   Filter: category='programming'")
        
        results, result_nodes = store.search(
            collection_name=collection_name,
            query=query,
            distance_type="cosine",
            number=5,
            meta_data_filters=programming_filters
        )
        
        print(f"   Found {len(results)} results:")
        for idx, node in enumerate(result_nodes, 1):
            lang = node.metadata.custom.get('language', 'N/A')
            print(f"      {idx}. [Language: {lang}] {node.content}")
        
        # Add more nodes
        print("\n8. Adding more nodes to existing collection...")
        new_documents = [
            {
                "content": "Kubernetes orchestrates container deployment and management.",
                "metadata": {"category": "devops", "topic": "orchestration", "difficulty": "advanced"}
            },
            {
                "content": "React is a JavaScript library for building user interfaces.",
                "metadata": {"category": "programming", "language": "javascript", "difficulty": "intermediate"}
            }
        ]
        
        new_nodes = []
        for idx, doc in enumerate(new_documents):
            metadata = NodeMetadata(
                source_file_uuid="additional_docs",
                position=idx,
                custom=doc["metadata"]
            )
            node = Node(content=doc["content"], metadata=metadata)
            new_nodes.append(node)
        
        store.add_nodes(collection_name, new_nodes)
        print(f"   ✓ Added {len(new_nodes)} new nodes")
        
        # Verify addition
        info = store.get_collection_info(collection_name)
        print(f"   ✓ Updated node count: {info['node_count']}")
        
        # Delete specific nodes
        print("\n9. Deleting nodes...")
        # Get first 2 node IDs
        search_results, search_nodes = store.search(
            collection_name=collection_name,
            query="test",
            number=2
        )
        
        node_ids_to_delete = [node.metadata.node_id for node in search_nodes[:2]]
        print(f"   Deleting node IDs: {node_ids_to_delete}")
        
        store.delete_nodes(collection_name, node_ids_to_delete)
        print(f"   ✓ Deleted {len(node_ids_to_delete)} nodes")
        
        # Verify deletion
        info = store.get_collection_info(collection_name)
        print(f"   ✓ Updated node count: {info['node_count']}")
        
        # List all collections
        print("\n10. Listing all collections...")
        collections = store.list_collections()
        print(f"   Found {len(collections)} collection(s):")
        for coll_name in collections:
            print(f"      - {coll_name}")
        
        # Compare with FAISS
        print("\n11. PostgreSQL vs FAISS comparison...")
        print("\n   PostgreSQL + pgvector:")
        print("   + Pros:")
        print("      - Persistent storage (survives restarts)")
        print("      - ACID transactions")
        print("      - Hybrid search (vector + full-text)")
        print("      - Built-in re-ranking")
        print("      - Scalable for production")
        print("   - Cons:")
        print("      - Requires database setup")
        print("      - Additional infrastructure")
        print("      - Network latency for remote DB")
        
        print("\n   FAISS:")
        print("   + Pros:")
        print("      - No database required")
        print("      - Extremely fast in-memory search")
        print("      - Easy to get started")
        print("   - Cons:")
        print("      - In-memory only (needs manual persistence)")
        print("      - No built-in transaction support")
        print("      - Limited filtering capabilities")
        
        print("\n" + "=" * 60)
        print("PostgreSQL Vector Store Example Completed!")
        print("=" * 60)
        
        # Cleanup (optional - comment out to keep data)
        print("\n⚠️  Cleanup: To delete the collection, uncomment the following:")
        print("   # store.delete_collection(collection_name)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check PostgreSQL is running")
        print("2. Verify pgvector extension is installed")
        print("3. Ensure Django migrations are up to date")
        print("4. Check database connection settings")
        raise


if __name__ == "__main__":
    main()


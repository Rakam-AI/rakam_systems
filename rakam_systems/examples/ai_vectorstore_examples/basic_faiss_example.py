"""
Basic FAISS Vector Store Example

This example demonstrates:
1. Creating a FAISS-based vector store
2. Adding documents as nodes
3. Performing semantic search
4. Managing collections
"""

from rakam_systems.ai_vectorstore.components.vectorstore.faiss_vector_store import FaissStore
from rakam_systems.ai_vectorstore.core import Node, NodeMetadata, VSFile
import os
from typing import List


def create_sample_documents() -> List[Node]:
    """Create sample documents for demonstration."""
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Vector databases store and retrieve data based on semantic similarity.",
        "FAISS is a library for efficient similarity search and clustering.",
        "Embeddings represent text as high-dimensional vectors.",
        "Transformers revolutionized natural language processing tasks.",
    ]
    
    nodes = []
    for idx, doc in enumerate(documents):
        metadata = NodeMetadata(
            source_file_uuid="sample_doc_001",
            position=idx,
            custom={"category": "AI/ML", "index": idx}
        )
        node = Node(content=doc, metadata=metadata)
        nodes.append(node)
    
    return nodes


def main():
    print("=" * 60)
    print("Basic FAISS Vector Store Example")
    print("=" * 60)
    
    # Initialize FAISS store
    # Using local embedding model (Snowflake Arctic)
    print("\n1. Initializing FAISS store...")
    store = FaissStore(
        name="demo_faiss_store",
        base_index_path="./faiss_demo_indexes",
        embedding_model="Snowflake/snowflake-arctic-embed-m",
        initialising=True,  # Don't load existing stores
        use_embedding_api=False  # Use local model
    )
    print("   ✓ FAISS store initialized")
    
    # Create sample documents
    print("\n2. Creating sample documents...")
    nodes = create_sample_documents()
    print(f"   ✓ Created {len(nodes)} sample nodes")
    
    # Create a collection from nodes
    print("\n3. Creating collection 'ai_ml_docs'...")
    collection_name = "ai_ml_docs"
    store.create_collection_from_nodes(collection_name, nodes)
    print(f"   ✓ Collection '{collection_name}' created with {len(nodes)} nodes")
    
    # Perform semantic search
    print("\n4. Performing semantic searches...")
    queries = [
        "What is machine learning?",
        "Tell me about neural networks",
        "How do vector databases work?"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        results, result_nodes = store.search(
            collection_name=collection_name,
            query=query,
            distance_type="cosine",
            number=3
        )
        
        print(f"   Top {len(results)} results:")
        for idx, (node_id, (metadata, content, distance)) in enumerate(results.items(), 1):
            print(f"      {idx}. [Distance: {distance:.4f}] {content[:60]}...")
    
    # Add more nodes to existing collection
    print("\n5. Adding more nodes to the collection...")
    new_documents = [
        "Reinforcement learning trains agents through rewards and penalties.",
        "Computer vision enables machines to interpret visual information."
    ]
    
    new_nodes = []
    for idx, doc in enumerate(new_documents):
        metadata = NodeMetadata(
            source_file_uuid="sample_doc_002",
            position=idx,
            custom={"category": "AI/ML", "index": len(nodes) + idx}
        )
        node = Node(content=doc, metadata=metadata)
        new_nodes.append(node)
    
    store.add_nodes(collection_name, new_nodes)
    print(f"   ✓ Added {len(new_nodes)} new nodes")
    
    # Search again with updated collection
    print("\n6. Searching updated collection...")
    query = "How do machines learn from rewards?"
    print(f"   Query: '{query}'")
    results, result_nodes = store.search(
        collection_name=collection_name,
        query=query,
        distance_type="cosine",
        number=2
    )
    
    print(f"   Top {len(results)} results:")
    for idx, (node_id, (metadata, content, distance)) in enumerate(results.items(), 1):
        print(f"      {idx}. [Distance: {distance:.4f}] {content}")
    
    # Delete some nodes
    print("\n7. Deleting nodes from collection...")
    # Delete the first 2 nodes
    node_ids_to_delete = [0, 1]
    store.delete_nodes(collection_name, node_ids_to_delete)
    print(f"   ✓ Deleted {len(node_ids_to_delete)} nodes")
    
    # Verify deletion
    print("\n8. Verifying deletion with search...")
    query = "What is Python?"
    results, result_nodes = store.search(
        collection_name=collection_name,
        query=query,
        distance_type="cosine",
        number=3
    )
    print(f"   Found {len(results)} results after deletion")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    # Cleanup
    print("\nCleaning up demo files...")
    import shutil
    if os.path.exists("./faiss_demo_indexes"):
        shutil.rmtree("./faiss_demo_indexes")
    print("✓ Cleanup complete")


if __name__ == "__main__":
    main()


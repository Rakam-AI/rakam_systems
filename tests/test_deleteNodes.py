from rakam_systems.core import Node, NodeMetadata, VSFile
from rakam_systems.components.vector_search.vector_store import VectorStore

# Define the base directory where the vector store index will be saved
BASE_INDEX_PATH = "vector_store_index_1"

# Define the embedding model to be used
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Step 1: Initialize the vector store
print("\n=== Step 1: Initializing the Vector Store ===")
vector_store = VectorStore(base_index_path=BASE_INDEX_PATH, embedding_model=EMBEDDING_MODEL)
print(f"Vector store initialized with base index path: {BASE_INDEX_PATH}\n")

# Step 2: Create a collection of nodes and store them in the vector store
print("\n=== Step 2: Creating a Collection and Storing Nodes ===")
collection_name = "sample_collection_0"

# Sample content for nodes
texts = [
    "This is the first document.",
    "Here is another document.",
    "Final document in the list."
]

# Create nodes with content and metadata
nodes = [
    Node(content=text, metadata=NodeMetadata(source_file_uuid=f"file_{i}", position=i))
    for i, text in enumerate(texts)
]

# Create a VSFile that contains the nodes
vs_file = VSFile(file_path="sample_file")
vs_file.nodes = nodes

# Create a collection from the VSFile in the vector store
vector_store.create_collection_from_files(collection_name=collection_name, files=[vs_file])
print(f"Collection '{collection_name}' created successfully!\n")

# Step 3: Perform a search in the created collection
print("\n=== Step 3: Searching the Collection ===")
query = "first document"

# Search for the query in the collection
results, _ = vector_store.search(collection_name=collection_name, query=query, number=3)

# Display search results
print(f"Search Results for query '{query}':")
for id_, (metadata, suggestion_text, distance) in results.items():
    print(f"ID: {id_}, Suggestion: {suggestion_text}, Metadata: {metadata}, Distance: {distance}")
print("\n")

# Step 4: Add new nodes to the existing collection
print("\n=== Step 4: Adding New Nodes to the Collection ===")
new_texts = ["New document added.", "Extra content for testing."]

# Create new nodes with content and metadata
new_nodes = [
    Node(content=text, metadata=NodeMetadata(source_file_uuid=f"repeated_new_file_{i}", position=i))
    for i, text in enumerate(new_texts)
]

# Add the new nodes to the collection
vector_store.add_nodes(collection_name=collection_name, nodes=new_nodes)
print(f"New nodes added to the collection '{collection_name}' successfully!\n")

# Step 5: Search the collection again after adding new nodes
print("\n=== Step 5: Searching the Collection After Adding New Nodes ===")
new_query = "new"
results, _ = vector_store.search(collection_name=collection_name, query=new_query, number=2)

# Display updated search results
print(f"Search Results for query '{new_query}':")
for id_, (metadata, suggestion_text, distance) in results.items():
    print(f"ID: {id_}, Suggestion: {suggestion_text}, Metadata: {metadata}, Distance: {distance}")
print("\n")

# Step 6: Delete specific nodes from the collection
print("\n=== Step 6: Deleting Nodes from the Collection ===")
node_ids_to_delete = [0, 1, 2, 3, 4, 5, 6]  

# Delete the nodes with the specified IDs
vector_store.delete_nodes(collection_name=collection_name, node_ids=node_ids_to_delete)

# Step 7: Save the collection and vector store to disk
print("\n=== Step 7: Saving the Collection and Vector Store ===")
vector_store._save_collection(collection_name=collection_name)
print(f"Collection '{collection_name}' saved to disk successfully!\n")

# Step 8: Reload the vector store from disk
print("\n=== Step 8: Reloading the Vector Store ===")
vector_store.load_vector_store()
print("Vector store reloaded from disk.")


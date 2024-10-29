import pytest
import os
import sys
import os
import sys
# Add the root directory of rakam_systems to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import numpy as np
from rakam_systems.core import Node, NodeMetadata, VSFile
from rakam_systems.components.vector_search.vector_store import VectorStore

BASE_INDEX_PATH = "test_vector_store_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "test_collection"

@pytest.fixture(scope="function")
def vector_store():
    """Fixture to initialize and clean up the VectorStore for testing."""
    store = VectorStore(base_index_path=BASE_INDEX_PATH, embedding_model=EMBEDDING_MODEL)
    yield store
    # Cleanup after the test
    # if os.path.exists(BASE_INDEX_PATH):
    #     shutil.rmtree(BASE_INDEX_PATH)

def create_sample_vs_file() -> VSFile:
    """Helper function to create a sample VSFile with nodes."""
    texts = ["First document.", "Second document.", "Third document."]
    nodes = [
        Node(content=text, metadata=NodeMetadata(source_file_uuid=f"file_{i}", position=i))
        for i, text in enumerate(texts)
    ]
    vs_file = VSFile(file_path="sample_file")
    vs_file.nodes = nodes
    return vs_file

def test_vector_store_initialization(vector_store):
    """Test vector store initialization."""
    assert os.path.exists(BASE_INDEX_PATH), "Base index path was not created."
    assert vector_store.embedding_model is not None, "Embedding model was not initialized."

def test_create_and_search_collection(vector_store):
    """Test creating a collection and performing a search."""
    vs_file = create_sample_vs_file()
    
    # Create collection from file
    vector_store.create_collection_from_files(collection_name=COLLECTION_NAME, files=[vs_file])
    
    # Perform a search in the created collection
    query = "First document"
    results, _ = vector_store.search(collection_name=COLLECTION_NAME, query=query, number=1)
    
    assert len(results) > 0, "Search returned no results."
    assert "First document" in results[next(iter(results))][1], "Expected document not found in results."

def test_add_and_search_nodes(vector_store):
    """Test adding nodes to a collection and searching them."""
    vs_file = create_sample_vs_file()
    
    # Create collection from file
    vector_store.create_collection_from_files(collection_name=COLLECTION_NAME, files=[vs_file])
    
    # Add new nodes
    new_nodes = [
        Node(content="New document added.", metadata=NodeMetadata(source_file_uuid="new_file", position=0)),
        Node(content="Extra document.", metadata=NodeMetadata(source_file_uuid="new_file", position=1))
    ]
    vector_store.add_nodes(collection_name=COLLECTION_NAME, nodes=new_nodes)
    
    # Perform a search for the new node
    query = "New document"
    results, _ = vector_store.search(collection_name=COLLECTION_NAME, query=query, number=1)
    
    assert len(results) > 0, "Search returned no results for the new node."
    assert "New document" in results[next(iter(results))][1], "Expected new document not found."

def test_delete_nodes(vector_store):
    """Test deleting nodes from a collection."""
    vs_file = create_sample_vs_file()
    
    # Create collection from file
    vector_store.create_collection_from_files(collection_name=COLLECTION_NAME, files=[vs_file])
    faiss_index = vector_store.collections.get(COLLECTION_NAME)["index"]
    # Perform a search before deleting nodes
    query = "First document"
    results, _ = vector_store.search(collection_name=COLLECTION_NAME, query=query, number=1)
    assert len(results) > 0, "Expected document not found before deletion."
    size_before_deletion = faiss_index.ntotal
    # Delete nodes
    node_ids = [0, 1, 2]
    vector_store.delete_nodes(collection_name=COLLECTION_NAME, node_ids=node_ids)
    size_after_deletion = faiss_index.ntotal
    diff = size_before_deletion - size_after_deletion
    # Perform a search after deleting nodes
    results, _ = vector_store.search(collection_name=COLLECTION_NAME, query=query, number=1)
    assert diff == 3, "Node was not deleted from the collection."

def test_save_and_load_vector_store(vector_store):
    """Test saving and reloading the vector store and collection."""
    vs_file = create_sample_vs_file()
    # Create collection and save it
    vector_store.create_collection_from_files(collection_name=COLLECTION_NAME, files=[vs_file])
    #vector_store._save_collection(collection_name=COLLECTION_NAME)
    # Verify that collection was saved
    saved_path = os.path.join(BASE_INDEX_PATH, COLLECTION_NAME)
    assert os.path.exists(saved_path), "Collection was not saved to disk."
    
    # Reload the vector store
    vector_store.load_vector_store()
    assert COLLECTION_NAME in vector_store.collections, "Collection was not reloaded from disk."
    
    # Perform a search to ensure data consistency after reloading
    query = "First document"
    results, _ = vector_store.search(collection_name=COLLECTION_NAME, query=query, number=1)
    assert len(results) > 0, "Search returned no results after reloading."




def test_delete_nonexistent_nodes(vector_store):
    """Test deleting non-existent nodes from a collection."""
    vs_file = create_sample_vs_file()
    
    # Create collection from file
    vector_store.create_collection_from_files(collection_name=COLLECTION_NAME, files=[vs_file])
    
    # Try to delete non-existent nodes
    nonexistent_node_ids = [999, 1000]
    vector_store.delete_nodes(collection_name=COLLECTION_NAME, node_ids=nonexistent_node_ids)
    
    # Perform a search to ensure valid nodes still exist
    query = "First document"
    results, _ = vector_store.search(collection_name=COLLECTION_NAME, query=query, number=1)
    assert len(results) > 0, "Search returned no results after trying to delete non-existent nodes."

def test_delete_partial_nodes(vector_store):
    """Test partially deleting nodes from a collection."""
    vs_file = create_sample_vs_file()
    
    # Create collection from file
    vector_store.create_collection_from_files(collection_name=COLLECTION_NAME, files=[vs_file])
    
    # Delete only the first two nodes
    vector_store.delete_nodes(collection_name=COLLECTION_NAME, node_ids=[0,1])
    
    # Search to ensure the third node is still present
    query = "Third document"
    results, _ = vector_store.search(collection_name=COLLECTION_NAME, query=query, number=1)
    assert len(results) > 0, "Search returned no results for the remaining node."
    assert "Third document" in results[next(iter(results))][1], "Remaining node was not found after partial deletion."

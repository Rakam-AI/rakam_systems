import sys
import os

# Add the root directory of rakam_systems to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import numpy as np
from rakam_systems.core import Node, NodeMetadata, VSFile
from rakam_systems.components.vector_search.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

@pytest.fixture
def vector_store_fixture(tmp_path):
    """
    Fixture to initialize the VectorStore with a temporary path and real embeddings.
    """
    # Initialize VectorStore
    base_index_path = tmp_path / "vector_store"
    vector_store = VectorStore(base_index_path=str(base_index_path), embedding_model="sentence-transformers/all-MiniLM-L6-v2", initialising=True)

    return vector_store

def test_init_vector_store(tmp_path):
    """
    Test the initialization of the VectorStore.
    """
    base_index_path = tmp_path / "vector_store"
    vector_store = VectorStore(base_index_path=str(base_index_path), embedding_model="sentence-transformers/all-MiniLM-L6-v2", initialising=True)

    # Check if the base index path was created
    assert os.path.exists(base_index_path)

def test_predict_embeddings(vector_store_fixture):
    """
    Test the prediction of embeddings for a given query.
    """
    query = "test query"
    embeddings = vector_store_fixture.predict_embeddings(query)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[1] == 384  # Shape should match the embedding size

def test_create_collection_from_files(vector_store_fixture, tmp_path):
    """
    Test the creation of a collection from VSFiles.
    """
    vs_file = VSFile(file_path="test_file_path")
    node_metadata = NodeMetadata(source_file_uuid="uuid1", position=1)
    node = Node(content="This is a test content", metadata=node_metadata)
    vs_file.nodes = [node]

    collection_name = "test_collection"
    vector_store_fixture.create_collection_from_files(collection_name=collection_name, files=[vs_file])

    # Check if collection exists
    assert collection_name in vector_store_fixture.collections
    store = vector_store_fixture.collections[collection_name]

    # Check if FAISS index and metadata were created
    assert "index" in store
    assert "category_index_mapping" in store
    assert "metadata_index_mapping" in store
    assert "nodes" in store
    assert len(store["nodes"]) == 1

def test_add_nodes_to_collection(vector_store_fixture):
    """
    Test adding nodes to an existing collection.
    """
    vs_file = VSFile(file_path="test_file_path")
    node_metadata = NodeMetadata(source_file_uuid="uuid1", position=1)
    node = Node(content="This is a test content", metadata=node_metadata)
    vs_file.nodes = [node]

    collection_name = "test_collection"
    vector_store_fixture.create_collection_from_files(collection_name=collection_name, files=[vs_file])

    # Add more nodes to the collection
    new_node_metadata = NodeMetadata(source_file_uuid="uuid2", position=2)
    new_node = Node(content="New test content", metadata=new_node_metadata)
    vector_store_fixture.add_nodes(collection_name=collection_name, nodes=[new_node])

    # Check if nodes were added to the collection
    store = vector_store_fixture.collections[collection_name]
    assert len(store["nodes"]) == 2

def test_delete_nodes_from_collection(vector_store_fixture):
    """
    Test deleting nodes from an existing collection.
    """
    vs_file = VSFile(file_path="test_file_path")
    node_metadata = NodeMetadata(source_file_uuid="uuid1", position=1)
    node = Node(content="This is a test content", metadata=node_metadata)
    vs_file.nodes = [node]

    collection_name = "test_collection"
    vector_store_fixture.create_collection_from_files(collection_name=collection_name, files=[vs_file])

    # Delete nodes from the collection
    vector_store_fixture.delete_nodes(collection_name=collection_name, node_ids=[0])

    # Check if nodes were deleted
    store = vector_store_fixture.collections[collection_name]
    assert len(store["nodes"]) == 0

def test_search_in_collection(vector_store_fixture):
    """
    Test searching in an existing collection.
    """
    vs_file = VSFile(file_path="test_file_path")
    node_metadata = NodeMetadata(source_file_uuid="uuid1", position=1)
    node = Node(content="This is a test content", metadata=node_metadata)
    vs_file.nodes = [node]

    collection_name = "test_collection"
    vector_store_fixture.create_collection_from_files(collection_name=collection_name, files=[vs_file])

    # Perform search
    results, suggested_nodes = vector_store_fixture.search(collection_name=collection_name, query="test")

    # Check if search results are returned
    assert isinstance(results, dict)
    assert len(suggested_nodes) > 0

def test_save_collection(vector_store_fixture, tmp_path):
    """
    Test saving the collection after changes.
    """
    vs_file = VSFile(file_path="test_file_path")
    node_metadata = NodeMetadata(source_file_uuid="uuid1", position=1)
    node = Node(content="This is a test content", metadata=node_metadata)
    vs_file.nodes = [node]

    collection_name = "test_collection"
    vector_store_fixture.create_collection_from_files(collection_name=collection_name, files=[vs_file])

    # Simulate saving collection
    vector_store_fixture._save_collection(collection_name=collection_name)

    # Check if collection files were saved to disk
    store_path = os.path.join(vector_store_fixture.base_index_path, collection_name)
    assert os.path.exists(os.path.join(store_path, "category_index_mapping.pkl"))
    assert os.path.exists(os.path.join(store_path, "metadata_index_mapping.pkl"))
    assert os.path.exists(os.path.join(store_path, "nodes.pkl"))
    assert os.path.exists(os.path.join(store_path, "index"))

import os
import pytest
import numpy as np
import faiss
import sys
# Add the root directory of rakam_systems to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer
from rakam_systems.core import VSFile, Node, NodeMetadata
from rakam_systems.components.vector_search.vector_store import VectorStore

# Directory for storing temporary test collections
TEST_BASE_PATH = "test_vector_store"

# Setup a test embedding model
TEST_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def vector_store():
    """
    Fixture to create and initialize a VectorStore instance for testing.
    """
    # Clean up any existing test directory
    if os.path.exists(TEST_BASE_PATH):
        for collection in os.listdir(TEST_BASE_PATH):
            collection_path = os.path.join(TEST_BASE_PATH, collection)
            if os.path.isdir(collection_path):
                for file in os.listdir(collection_path):
                    os.remove(os.path.join(collection_path, file))
                os.rmdir(collection_path)
        os.rmdir(TEST_BASE_PATH)

    # Initialize the VectorStore
    store = VectorStore(TEST_BASE_PATH, TEST_EMBEDDING_MODEL, initialising=True)
    return store


@pytest.fixture
def sample_nodes():
    """
    Fixture to create a list of sample nodes with metadata for testing.
    """
    nodes = [
        Node(
            content="This is a test content 1.",
            metadata=NodeMetadata(
                source_file_uuid="file1",
                position=0,
                custom={"type": "test"}
            )
        ),
        Node(
            content="Another test content 2.",
            metadata=NodeMetadata(
                source_file_uuid="file2",
                position=1,
                custom={"type": "example"}
            )
        ),
        Node(
            content="Additional content 3.",
            metadata=NodeMetadata(
                source_file_uuid="file3",
                position=2,
                custom={"type": "sample"}
            )
        ),
        Node(
            content="More test content 4.",
            metadata=NodeMetadata(
                source_file_uuid="file4",
                position=3,
                custom={"type": "example"}
            )
        )
    ]
    
    # Manually set the node_id attribute if necessary
    nodes[0].metadata.node_id = 0
    nodes[1].metadata.node_id = 1
    nodes[2].metadata.node_id = 2
    nodes[3].metadata.node_id = 3
    
    return nodes

def test_create_and_search_collection(vector_store, sample_nodes):
    collection_name = "test_collection"
    vector_store.create_collection_from_nodes(collection_name, sample_nodes)
    assert collection_name in vector_store.collections, "Collection not created successfully."

    query = "test content"
    results, nodes = vector_store.search(collection_name, query, number=2)

    print("Search results:", results)
    print("Returned nodes:", nodes)

    assert results, "No results found."
    actual_results = {k: v for k, v in results.items() if "No suggestion" not in v[1]}
    print("Actual results after filtering:", actual_results)
    assert len(actual_results) == 2, f"Expected 2 results, got {len(actual_results)}."
    assert len(nodes) == 2, "Node suggestions count mismatch."

def test_search_with_metadata_filter(vector_store, sample_nodes):
    """
    Test case for searching within a collection with metadata filters.
    """
    collection_name = "filtered_collection"

    # Create a collection with sample nodes
    vector_store.create_collection_from_nodes(collection_name, sample_nodes)
    assert collection_name in vector_store.collections, "Collection not created successfully."

    # Define a query and metadata filters
    query = "test content"
    meta_data_filters = [0, 1]  # Filter by node IDs
    results, nodes = vector_store.search(
        collection_name, query, meta_data_filters=meta_data_filters, number=3
    )

    # Ensure that the results match the filters
    actual_results = {k: v for k, v in results.items() if "No suggestion" not in v[1]}

    # Assertions to validate correctness
    assert nodes[0].content in ["This is a test content 1.","Another test content 2."]
    assert nodes[0].content not in ["Additional content 3.","More test content 4."]
    assert len(actual_results) <= 2, f"Expected at most 2 results, got {len(actual_results)}."
    assert len(nodes) == len(actual_results), "Returned nodes count mismatch with results."
    assert all(
        node.metadata.node_id in meta_data_filters for node in nodes
    ), "Search results do not match the metadata filter."


def test_search_empty_collection(vector_store):
    """
    Test case for searching in a collection that does not exist.
    """
    collection_name = "non_existent_collection"

    # Attempting to search in a non-existent collection should raise a ValueError
    with pytest.raises(ValueError, match=f"No store found with name: {collection_name}"):
        vector_store.search(collection_name, query="test")


def test_predict_embeddings(vector_store):
    """
    Test the predict_embeddings function to ensure embeddings are generated correctly.
    """
    query = "test embedding generation"
    embedding = vector_store.predict_embeddings(query)
    assert isinstance(embedding, np.ndarray), "The output is not a numpy array."
    assert embedding.shape[1] > 0, "Embedding vector has invalid dimensions."


if __name__ == "__main__":
    pytest.main()

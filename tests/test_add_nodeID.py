import os
import pytest
import shutil
from rakam_systems.components.vector_search.vector_store import VectorStore, VSFile, Node, NodeMetadata


@pytest.fixture
def setup_vector_store():
    # Create a temporary directory for the test
    base_index_path = "test_data"
    embedding_model = "paraphrase-MiniLM-L6-v2"
    
    # Ensure the test directory is clean before starting
    if os.path.exists(base_index_path):
        shutil.rmtree(base_index_path)
    
    os.makedirs(base_index_path)
    
    # Initialize the VectorStore
    vs = VectorStore(base_index_path=base_index_path, embedding_model=embedding_model)
    yield vs

    # Clean up after the test
    shutil.rmtree(base_index_path)


def test_create_collection_from_files(setup_vector_store):
    vs = setup_vector_store

    # Define example VSFiles with nodes and metadata
    vsfiles_example = [
        VSFile("data/1.txt"),
        VSFile("data/2.txt"),
    ]
    
    # Nodes for the first file
    nodes_1 = [
        Node("This is a test", NodeMetadata("1", 1)),
        Node("This is another test", NodeMetadata("1", 2)),
    ]
    # Nodes for the second file
    nodes_2 = [
        Node("This is a test", NodeMetadata("2", 1)),
        Node("This is another test", NodeMetadata("2", 2)),
    ]
    
    # Assign nodes to the VSFiles
    vsfiles_example[0].nodes = nodes_1
    vsfiles_example[1].nodes = nodes_2

    # Create the collection from the VSFiles
    collection_name = "test_collection"
    vs.create_collection_from_files(collection_name, vsfiles_example)

    # Validate that the nodes have been assigned the correct node_id
    nodes = vs.collections[collection_name]["nodes"]
    assert nodes[0].metadata.node_id == 0, "Node ID for the first node should be 0"
    assert nodes[1].metadata.node_id == 1, "Node ID for the second node should be 1"
    assert nodes[2].metadata.node_id == 0, "Node ID for the third node should be 0"
    assert nodes[3].metadata.node_id == 1, "Node ID for the fourth node should be 1"

    print(f"Assigned node IDs: {[node.metadata.node_id for node in nodes]}")


if __name__ == "__main__":
    pytest.main()

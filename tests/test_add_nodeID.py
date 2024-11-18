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
    ]
    
    node_1 = Node("Node content 1", NodeMetadata(source_file_uuid=vsfiles_example[0].uuid, position=0, custom={"key": "value"}))
    node_2 = Node("Node content 2", NodeMetadata(source_file_uuid=vsfiles_example[0].uuid, position=1, custom={"key": "value"}))
    node_3 = Node("Node content 3", NodeMetadata(source_file_uuid=vsfiles_example[0].uuid, position=2, custom={"key": "value"}))
    node_4 = Node("Node content 4", NodeMetadata(source_file_uuid=vsfiles_example[0].uuid, position=3, custom={"key": "value"}))

    # Assign nodes to the VSFiles
    vsfiles_example[0].nodes = [
        node_1,
        node_2,
        node_3,
        node_4,
    ]

    # Create the collection from the VSFiles
    collection_name = "test_collection"
    vs.create_collection_from_files(collection_name, vsfiles_example)

    # Validate that the node_ids in the nodes' metadata have been assigned correctly
    nodes = vs.collections[collection_name]["nodes"]
    assert nodes[0].metadata.node_id == 0, "Node ID for the first node should be 0"
    assert nodes[1].metadata.node_id == 1, "Node ID for the second node should be 1"
    assert nodes[2].metadata.node_id == 2, "Node ID for the third node should be 0"
    assert nodes[3].metadata.node_id == 3, "Node ID for the fourth node should be 1"
    assert len(nodes) == 4, "Number of nodes in the collection should be 4"

    # Validate the node_ids in the metadata has been assigned correctly
    metadata = [node.metadata for node in nodes]
    assert metadata[0].node_id == 0, "Node ID for the first node should be 0"
    assert metadata[1].node_id == 1, "Node ID for the second node should be 1"
    assert metadata[2].node_id == 2, "Node ID for the third node should be 0"
    assert metadata[3].node_id == 3, "Node ID for the fourth node should be 1"

    print(f"Assigned node IDs: {[node.metadata.node_id for node in nodes]}")


def test_create_collection_from_nodes(setup_vector_store):
    vs = setup_vector_store

    # Define a list of nodes with content and metadata
    nodes = [
        Node("Node content 1", NodeMetadata(source_file_uuid="file_1", position=1, custom={"key": "value"})),
        Node("Node content 2", NodeMetadata(source_file_uuid="file_1", position=2, custom={"key": "value"})),
        Node("Node content 3", NodeMetadata(source_file_uuid="file_2", position=1, custom={"key": "another"})),
    ]

    collection_name = "nodes_collection"
    
    # Create a collection from nodes
    vs.create_collection_from_nodes(collection_name, nodes)

    # Validate the collection has been created
    assert collection_name in vs.collections, f"Collection '{collection_name}' should exist."

    # Validate the nodes in the collection
    created_nodes = vs.collections[collection_name]["nodes"]
    assert len(created_nodes) == len(nodes), "Number of nodes in the collection should match the input nodes."

    # Check if the metadata is correctly stored
    for i, node in enumerate(created_nodes):
        assert node.content == nodes[i].content, f"Content mismatch for node {i}"
        assert node.metadata.source_file_uuid == nodes[i].metadata.source_file_uuid, f"Metadata mismatch for node {i}"
        assert node.metadata.position == nodes[i].metadata.position, f"Position mismatch for node {i}"
        assert node.metadata.custom == nodes[i].metadata.custom, f"Custom metadata mismatch for node {i}"
        assert node.metadata.node_id == i, f"Node ID mismatch for node {i}"

    # Check if the node_ids have been assigned correctly in the metadata
    metadata = [node.metadata for node in created_nodes]
    assert metadata[0].node_id == 0, "Node ID for the first node should be 0"
    assert metadata[1].node_id == 1, "Node ID for the second node should be 1"
    assert metadata[2].node_id == 2, "Node ID for the third node should be 2"
    
    print(f"Successfully created collection with nodes: {[node.content for node in created_nodes]}")


if __name__ == "__main__":
    pytest.main()

import os
import sys
import pytest
import shutil
import logging
from rakam_systems.components.vector_search.vector_store import VectorStore, VSFile, Node, NodeMetadata

# Add the root directory of rakam_systems to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    
    # Assign nodes to the VSFiles
    vsfiles_example[0].nodes = [
    Node("Node content 1", NodeMetadata(source_file_uuid=vsfiles_example[0].uuid, position=0, custom={"key": "value"})),
    Node("Node content 2", NodeMetadata(source_file_uuid=vsfiles_example[0].uuid, position=1, custom={"key": "value"})),
    Node("Node content 3", NodeMetadata(source_file_uuid=vsfiles_example[0].uuid, position=2, custom={"key": "value"})),
    Node("Node content 4", NodeMetadata(source_file_uuid=vsfiles_example[0].uuid, position=3, custom={"key": "value"})),
    ]

    vsfiles_example[1].nodes = [
    Node("Node content 1", NodeMetadata(source_file_uuid=vsfiles_example[1].uuid, position=0, custom={"key": "value"})),
    Node("Node content 2", NodeMetadata(source_file_uuid=vsfiles_example[1].uuid, position=1, custom={"key": "value"})),
    Node("Node content 3", NodeMetadata(source_file_uuid=vsfiles_example[1].uuid, position=2, custom={"key": "value"})),
    Node("Node content 4", NodeMetadata(source_file_uuid=vsfiles_example[1].uuid, position=3, custom={"key": "value"})),
    ]

    # Create the collection from the VSFiles
    collection_name = "test_collection"
    vs.create_collection_from_files(collection_name, vsfiles_example)

    # Validate that the node_ids in the nodes' metadata have been assigned correctly
    nodes = vs.collections[collection_name]["nodes"]
    assert nodes[0].metadata.node_id == 0, "Node ID for the first node should be 0"
    assert nodes[1].metadata.node_id == 1, "Node ID for the second node should be 1"
    assert nodes[2].metadata.node_id == 2, "Node ID for the third node should be 2"
    assert nodes[3].metadata.node_id == 3, "Node ID for the fourth node should be 3"
    assert nodes[4].metadata.node_id == 4, "Node ID for the fifth node should be 4"
    assert nodes[5].metadata.node_id == 5, "Node ID for the sixth node should be 5"
    assert nodes[6].metadata.node_id == 6, "Node ID for the seventh node should be 6"
    assert nodes[7].metadata.node_id == 7, "Node ID for the eighth node should be 7"
    print(f"length of nodes: {len(nodes)}")
    assert len(nodes) == 8, "Number of nodes in the collection should be 8"

    assert nodes[0].content == nodes[4].content, "Content of the first and fifth nodes should be the same"

    # Validate the node_ids in the metadata has been assigned correctly
    metadata = [node.metadata for node in nodes]
    assert metadata[0].node_id == 0, "Node ID for the first node should be 0"
    assert metadata[1].node_id == 1, "Node ID for the second node should be 1"
    assert metadata[2].node_id == 2, "Node ID for the third node should be 2"
    assert metadata[3].node_id == 3, "Node ID for the fourth node should be 3"
    assert metadata[4].node_id == 4, "Node ID for the fifth node should be 4"
    assert metadata[5].node_id == 5, "Node ID for the sixth node should be 5"
    assert metadata[6].node_id == 6, "Node ID for the seventh node should be 6"
    assert metadata[7].node_id == 7, "Node ID for the eighth node should be 7"

    print(f"Assigned node IDs: {[node.metadata.node_id for node in nodes]}")

    # Do search on the collection
    query = "Node content 1"
    results, nodes = vs.search(collection_name, query, number=2, meta_data_filters= [0, 1])
    print(f"Search results: {results}")
    print(f"Nodes: {[node.content for node in nodes]}")

    assert nodes[0].metadata.node_id == 0, "Node ID for the first node should be 0"
    assert nodes[1].metadata.node_id == 1, "Node ID for the second node should be 1"


def test_create_collection_from_nodes(setup_vector_store):
    vs = setup_vector_store

    # Define a list of nodes with content and metadata
    nodes = [
        Node("Node content 1", NodeMetadata(source_file_uuid="file_1", position=1, custom={"key": "value"})),
        Node("Node content 2", NodeMetadata(source_file_uuid="file_1", position=2, custom={"key": "value"})),
        Node("Node content 1", NodeMetadata(source_file_uuid="file_2", position=3, custom={"key": "another"})),
        Node("Node content 2", NodeMetadata(source_file_uuid="file_2", position=1, custom={"key": "another"})),
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
    assert metadata[3].node_id == 3, "Node ID for the fourth node should be 3"

    
    print(f"Successfully created collection with nodes: {[node.content for node in created_nodes]}")

    nodes = vs.collections[collection_name]["nodes"]
    assert nodes[0].metadata.node_id == 0, "Node ID for the first node should be 0"
    assert nodes[1].metadata.node_id == 1, "Node ID for the second node should be 1"
    assert nodes[2].metadata.node_id == 2, "Node ID for the third node should be 2"
    assert nodes[3].metadata.node_id == 3, "Node ID for the fourth node should be 3"

    assert nodes[0].content == nodes[2].content, "Content of the first and third nodes should be the same"

    # Do search on the collection
    query = "Node content 1"
    results, nodes = vs.search(collection_name, query, number=2, meta_data_filters= [0, 1])
    print(f"Search results: {results}")
    print(f"Nodes: {[node.content for node in nodes]}")

    assert nodes[0].metadata.node_id == 0, "Node ID for the first node should be 0"
    assert nodes[1].metadata.node_id == 1, "Node ID for the second node should be 1"


if __name__ == "__main__":
    pytest.main()

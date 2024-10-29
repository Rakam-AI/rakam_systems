import pytest
import mimetypes
from rakam_systems.core import VSFile, NodeMetadata, Node
import uuid


def test_vsfile_initialization():
    # Test initialization of VSFile
    file_path = "/path/to/sample.pdf"
    vs_file = VSFile(file_path)

    assert isinstance(vs_file.uuid, uuid.UUID), "UUID should be auto-generated"
    assert vs_file.file_path == file_path
    assert vs_file.file_name == "sample.pdf"
    assert vs_file.mime_type == mimetypes.guess_type(file_path)[0], "Mime type should be guessed"
    assert vs_file.nodes == [], "Nodes should be an empty list initially"
    assert not vs_file.processed, "Processed should be False initially"


def test_node_metadata_initialization():
    # Test initialization of NodeMetadata
    metadata = NodeMetadata(source_file_uuid="1234", position=1, custom={"author": "test_user"})

    assert metadata.source_file_uuid == "1234"
    assert metadata.position == 1
    assert metadata.custom == {"author": "test_user"}
    assert metadata.node_id is None, "Node ID should be None by default"


def test_node_metadata_str():
    # Test __str__ method of NodeMetadata
    metadata = NodeMetadata(source_file_uuid="1234", position=1, custom={"author": "test_user"})
    expected_str = (
        "NodeMetadata(node_id=None, source_file_uuid='1234', position=1, custom={ author: test_user })"
    )
    assert str(metadata) == expected_str


def test_node_initialization():
    # Test initialization of Node
    metadata = NodeMetadata(source_file_uuid="1234", position=1, custom={"author": "test_user"})
    node = Node(content="This is a test node content", metadata=metadata)

    assert node.content == "This is a test node content"
    assert node.metadata == metadata
    assert node.embedding is None, "Embedding should be None initially"



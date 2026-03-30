import pytest
from rakam_systems_core.vs_core import VSFile, NodeMetadata, Node


def test_vsfile_init():
    vsfile = VSFile("/some/path/document.pdf")
    assert vsfile.file_path == "/some/path/document.pdf"
    assert vsfile.file_name == "document.pdf"
    assert vsfile.nodes == []
    assert vsfile.processed is False
    assert vsfile.uuid is not None


def test_vsfile_mime_type_pdf():
    vsfile = VSFile("/docs/report.pdf")
    assert vsfile.mime_type == "application/pdf"


def test_vsfile_mime_type_text():
    vsfile = VSFile("/docs/notes.txt")
    assert vsfile.mime_type == "text/plain"


def test_vsfile_mime_type_unknown():
    vsfile = VSFile("/docs/file.unknownextension123")
    assert vsfile.mime_type is None


def test_vsfile_unique_uuids():
    vsfile1 = VSFile("/a.pdf")
    vsfile2 = VSFile("/b.pdf")
    assert vsfile1.uuid != vsfile2.uuid


def test_node_metadata_init():
    meta = NodeMetadata(source_file_uuid="uuid-123", position=0)
    assert meta.source_file_uuid == "uuid-123"
    assert meta.position == 0
    assert meta.node_id is None
    assert meta.custom is None


def test_node_metadata_with_custom():
    meta = NodeMetadata(
        source_file_uuid="uuid-abc",
        position=5,
        custom={"page": 1, "section": "intro"}
    )
    assert meta.custom["page"] == 1
    assert meta.custom["section"] == "intro"


def test_node_metadata_str():
    meta = NodeMetadata(source_file_uuid="uuid-111", position=2)
    meta.node_id = 10
    s = str(meta)
    assert "uuid-111" in s
    assert "10" in s
    assert "position=2" in s


def test_node_metadata_str_with_custom():
    meta = NodeMetadata(
        source_file_uuid="uuid-222",
        position=3,
        custom={"key": "value"}
    )
    s = str(meta)
    assert "key" in s
    assert "value" in s


def test_node_init():
    meta = NodeMetadata(source_file_uuid="uuid-xyz", position=0)
    node = Node(content="This is some text.", metadata=meta)
    assert node.content == "This is some text."
    assert node.metadata is meta
    assert node.embedding is None


def test_node_str():
    meta = NodeMetadata(source_file_uuid="abc", position=0)
    node = Node(content="Hello world, this is a test node", metadata=meta)
    s = str(node)
    assert "Node" in s
    assert "Hello world" in s


def test_node_str_long_content():
    meta = NodeMetadata(source_file_uuid="abc", position=0)
    long_content = "A" * 100
    node = Node(content=long_content, metadata=meta)
    s = str(node)
    assert "..." in s


def test_node_embedding_can_be_set():
    meta = NodeMetadata(source_file_uuid="abc", position=0)
    node = Node(content="text", metadata=meta)
    node.embedding = [0.1, 0.2, 0.3]
    assert node.embedding == [0.1, 0.2, 0.3]

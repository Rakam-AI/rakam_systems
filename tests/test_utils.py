import pytest
from rakam_systems.core import VSFile, NodeMetadata, Node
from rakam_systems.ingestion.utils import (
    llama_documents_to_VSFiles,
    llama_documents_to_VSFile,
    parsed_url_to_VSFile,
)


class MockDocument:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


@pytest.fixture
def llama_documents():
    return [
        MockDocument("Document 1 Text", {"file_name": "file1.pdf"}),
        MockDocument("Document 2 Text", {"file_name": "file1.pdf"}),
        MockDocument("Document 3 Text", {"file_name": "file2.pdf"}),
    ]


def test_llama_documents_to_VSFiles(llama_documents):
    vs_files = llama_documents_to_VSFiles(llama_documents)
    assert len(vs_files) == 2  # two distinct files
    assert len(vs_files[0].nodes) == 2  # file1.pdf has 2 nodes
    assert len(vs_files[1].nodes) == 1  # file2.pdf has 1 node
    assert vs_files[0].nodes[0].content == "Document 1 Text"
    assert vs_files[0].nodes[1].content == "Document 2 Text"
    assert vs_files[1].nodes[0].content == "Document 3 Text"


def test_llama_documents_to_VSFile(llama_documents):
    vs_file = llama_documents_to_VSFile(llama_documents[:2])

    assert len(vs_file.nodes) == 2
    assert vs_file.nodes[0].content == "Document 1 Text"
    assert vs_file.nodes[1].content == "Document 2 Text"


def test_parsed_url_to_VSFile():
    url = "https://www.example.com"
    content = "Sample extracted content"
    vs_file = parsed_url_to_VSFile(url, content, {"custom_meta": "meta_value"})

    assert vs_file.file_path == url
    assert len(vs_file.nodes) == 1
    assert vs_file.nodes[0].content == content
    assert vs_file.nodes[0].metadata.custom["custom_meta"] == "meta_value"

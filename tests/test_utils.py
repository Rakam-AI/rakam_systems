import pytest
from rakam_systems.core import VSFile, NodeMetadata, Node
from rakam_systems.ingestion.utils import (
    parsed_url_to_VSFile,
)


class MockDocument:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata

def test_parsed_url_to_VSFile():
    url = "https://www.example.com"
    content = "Sample extracted content"
    vs_file = parsed_url_to_VSFile(url, content, {"custom_meta": "meta_value"})

    assert vs_file.file_path == url
    assert len(vs_file.nodes) == 1
    assert vs_file.nodes[0].content == content
    assert vs_file.nodes[0].metadata.custom["custom_meta"] == "meta_value"

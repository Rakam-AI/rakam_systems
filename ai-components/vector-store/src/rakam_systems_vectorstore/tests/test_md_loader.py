from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rakam_systems_vectorstore.components.loader.md_loader import MdLoader, create_md_loader
from rakam_systems_vectorstore.core import Node, VSFile


@pytest.fixture
def loader():
    with patch("rakam_systems_vectorstore.components.loader.md_loader.AdvancedChunker") as mock_cls:
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = [{"text": "chunk1"}, {"text": "chunk2"}]
        mock_cls.return_value = mock_chunker
        yield MdLoader()


@pytest.fixture
def md_file(tmp_path):
    content = "# Title\n\nSome content here.\n\n## Section 2\n\nMore content."
    p = tmp_path / "test.md"
    p.write_text(content)
    return str(p)


@pytest.fixture
def md_file_with_frontmatter(tmp_path):
    content = "---\ntitle: My Doc\nauthor: Alice\n---\n# Title\n\nContent here."
    p = tmp_path / "doc.md"
    p.write_text(content)
    return str(p)


def test_loader_init():
    with patch("rakam_systems_vectorstore.components.loader.md_loader.AdvancedChunker"):
        loader = MdLoader()
    assert loader.name == "md_loader"
    assert loader._split_by_headers is True
    assert loader._preserve_code_blocks is True
    assert loader._extract_frontmatter is True


def test_create_md_loader_factory():
    with patch("rakam_systems_vectorstore.components.loader.md_loader.AdvancedChunker"):
        loader = create_md_loader(chunk_size=512, split_by_headers=False)
    assert loader._chunk_size == 512
    assert loader._split_by_headers is False


def test_load_as_text(loader, md_file):
    text = loader.load_as_text(md_file)
    assert isinstance(text, str)
    assert "Title" in text or "content" in text.lower()


def test_load_as_text_file_not_found(loader):
    with pytest.raises(FileNotFoundError):
        loader.load_as_text("/nonexistent/file.md")


def test_load_as_text_wrong_extension(loader, tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("some text")
    with pytest.raises(ValueError, match="not a Markdown file"):
        loader.load_as_text(str(f))


def test_load_as_text_with_path_object(loader, md_file):
    text = loader.load_as_text(Path(md_file))
    assert isinstance(text, str)


def test_load_as_text_strips_frontmatter(loader, md_file_with_frontmatter):
    text = loader.load_as_text(md_file_with_frontmatter)
    assert "---" not in text or "title:" not in text


def test_load_as_chunks(loader, md_file):
    chunks = loader.load_as_chunks(md_file)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, str)


def test_load_as_chunks_file_not_found(loader):
    with pytest.raises(FileNotFoundError):
        loader.load_as_chunks("/no/file.md")


def test_load_as_chunks_wrong_extension(loader, tmp_path):
    f = tmp_path / "test.html"
    f.write_text("<html>text</html>")
    with pytest.raises(ValueError):
        loader.load_as_chunks(str(f))


def test_load_as_nodes(loader, md_file):
    nodes = loader.load_as_nodes(md_file)
    assert isinstance(nodes, list)
    assert len(nodes) > 0
    for node in nodes:
        assert isinstance(node, Node)
        assert isinstance(node.content, str)
        assert node.metadata is not None


def test_load_as_nodes_with_custom_metadata(loader, md_file):
    nodes = loader.load_as_nodes(md_file, custom_metadata={"extra": "meta"})
    for node in nodes:
        assert node.metadata.custom["extra"] == "meta"


def test_load_as_nodes_positions_are_sequential(loader, md_file):
    nodes = loader.load_as_nodes(md_file)
    for idx, node in enumerate(nodes):
        assert node.metadata.position == idx


def test_load_as_vsfile(loader, md_file):
    vsfile = loader.load_as_vsfile(md_file)
    assert isinstance(vsfile, VSFile)
    assert vsfile.processed is True
    assert len(vsfile.nodes) > 0


def test_load_as_vsfile_not_found(loader):
    with pytest.raises(FileNotFoundError):
        loader.load_as_vsfile("/no/such/file.md")


def test_load_as_vsfile_wrong_extension(loader, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_text("data")
    with pytest.raises(ValueError):
        loader.load_as_vsfile(str(f))


def test_run_delegates_to_load_as_chunks(loader, md_file):
    result = loader.run(md_file)
    assert isinstance(result, list)


def test_is_md_file_extensions(loader):
    assert loader._is_md_file("doc.md") is True
    assert loader._is_md_file("doc.markdown") is True
    assert loader._is_md_file("doc.mdown") is True
    assert loader._is_md_file("doc.mkd") is True
    assert loader._is_md_file("doc.mkdn") is True
    assert loader._is_md_file("doc.txt") is False
    assert loader._is_md_file("doc.pdf") is False


def test_extract_headers(loader):
    content = "# Header 1\n\nText\n\n## Header 2\n\nMore text\n\n### Sub\n"
    headers = loader._extract_headers(content)
    assert len(headers) == 3
    assert headers[0]["level"] == 1
    assert headers[0]["text"] == "Header 1"
    assert headers[1]["level"] == 2
    assert headers[2]["level"] == 3


def test_extract_headers_empty(loader):
    headers = loader._extract_headers("No headers here.")
    assert headers == []


def test_frontmatter_to_text(loader):
    fm = {"title": "My Doc", "tags": ["a", "b"]}
    text = loader._frontmatter_to_text(fm)
    assert "title: My Doc" in text
    assert "tags: a, b" in text


def test_frontmatter_to_text_empty(loader):
    assert loader._frontmatter_to_text({}) == ""
    assert loader._frontmatter_to_text(None) == ""


def test_chunk_by_headers_empty(loader):
    result = loader._chunk_by_headers("")
    assert result == []


def test_chunk_by_headers_with_headers(loader, tmp_path):
    with patch("rakam_systems_vectorstore.components.loader.md_loader.AdvancedChunker") as mock_cls:
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = [{"text": "fallback"}]
        mock_cls.return_value = mock_chunker
        l = MdLoader()

    content = "# Header One\n\nFirst section content.\n\n## Header Two\n\nSecond section content."
    chunks = l._chunk_by_headers(content)
    assert len(chunks) >= 1


def test_get_frontmatter_none_initially():
    with patch("rakam_systems_vectorstore.components.loader.md_loader.AdvancedChunker"):
        loader = MdLoader()
    assert loader.get_frontmatter() is None


def test_get_headers_empty_initially():
    with patch("rakam_systems_vectorstore.components.loader.md_loader.AdvancedChunker"):
        loader = MdLoader()
    assert loader.get_headers() == []


def test_get_frontmatter_after_load(loader, md_file_with_frontmatter):
    loader.load_as_text(md_file_with_frontmatter)
    fm = loader.get_frontmatter()
    assert fm is not None
    assert fm.get("title") == "My Doc"


def test_get_headers_after_load(loader, md_file):
    loader.load_as_text(md_file)
    headers = loader.get_headers()
    assert isinstance(headers, list)

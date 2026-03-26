"""Tests for EmlLoader."""
from pathlib import Path
from unittest.mock import patch

import pytest

from rakam_systems_vectorstore.components.loader.eml_loader import (
    EmlLoader,
    create_eml_loader,
)
from rakam_systems_vectorstore.core import Node, VSFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plain_text_eml(
    tmp_path: Path,
    subject: str = "Test Email",
    body: str = "Hello World",
    from_addr: str = "sender@example.com",
    to_addr: str = "recipient@example.com",
) -> Path:
    """Create a minimal plain text .eml file."""
    content = (
        f"From: {from_addr}\r\n"
        f"To: {to_addr}\r\n"
        f"Subject: {subject}\r\n"
        f"Date: Mon, 01 Jan 2024 00:00:00 +0000\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n"
        f"\r\n"
        f"{body}\r\n"
    )
    p = tmp_path / "test.eml"
    p.write_bytes(content.encode("utf-8"))
    return p


def make_multipart_eml(tmp_path: Path, body: str = "Plain body") -> Path:
    """Create a minimal multipart .eml file."""
    boundary = "boundary123"
    content = (
        f"From: sender@example.com\r\n"
        f"To: recipient@example.com\r\n"
        f"Subject: Multipart Test\r\n"
        f"MIME-Version: 1.0\r\n"
        f'Content-Type: multipart/alternative; boundary="{boundary}"\r\n'
        f"\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n"
        f"\r\n"
        f"{body}\r\n"
        f"--{boundary}--\r\n"
    )
    p = tmp_path / "multipart.eml"
    p.write_bytes(content.encode("utf-8"))
    return p


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestEmlLoaderInit:
    def test_defaults(self):
        loader = EmlLoader()
        assert loader.config['chunk_size'] == 3000
        assert loader.config['chunk_overlap'] == 200
        assert loader.config['include_headers'] is True
        assert loader.config['extract_html'] is True

    def test_custom_config(self):
        loader = EmlLoader(
            config={
                "chunk_size": 1000,
                "chunk_overlap": 50,
                "include_headers": False,
                "extract_html": False,
            }
        )
        assert loader.config['chunk_size'] == 1000
        assert loader.config['chunk_overlap'] == 50
        assert loader.config['include_headers'] is False
        assert loader.config['extract_html'] is False

    def test_custom_name(self):
        loader = EmlLoader(name="my_eml_loader")
        assert loader.name == "my_eml_loader"


# ---------------------------------------------------------------------------
# _is_eml_file
# ---------------------------------------------------------------------------


class TestIsEmlFile:
    def test_eml_extension(self):
        loader = EmlLoader()
        assert loader.is_eml_file("test.eml") is True

    def test_eml_uppercase(self):
        loader = EmlLoader()
        assert loader.is_eml_file("test.EML") is True

    def test_non_eml_extensions(self):
        loader = EmlLoader()
        assert loader.is_eml_file("test.txt") is False
        assert loader.is_eml_file("test.msg") is False
        assert loader.is_eml_file("test.pdf") is False


# ---------------------------------------------------------------------------
# load_as_text
# ---------------------------------------------------------------------------


class TestLoadAsText:
    def test_basic_email_body(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path, body="Hello World!")
        loader = EmlLoader()
        text = loader.load_as_text(str(p))
        assert "Hello World!" in text

    def test_includes_headers_by_default(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path, subject="My Subject")
        loader = EmlLoader(config={"include_headers": True})
        text = loader.load_as_text(str(p))
        assert "Subject: My Subject" in text
        assert "From: sender@example.com" in text

    def test_excludes_headers_when_disabled(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path, subject="My Subject", body="Content here")
        loader = EmlLoader(config={"include_headers": False})
        text = loader.load_as_text(str(p))
        assert "Subject:" not in text
        assert "Content here" in text

    def test_file_not_found_raises(self):
        loader = EmlLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_as_text("/nonexistent/file.eml")

    def test_non_eml_file_raises(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("not an eml")
        loader = EmlLoader()
        with pytest.raises(ValueError):
            loader.load_as_text(str(f))

    def test_path_object_input(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path)
        loader = EmlLoader()
        text = loader.load_as_text(p)  # Pass Path object
        assert isinstance(text, str)

    def test_multipart_email(self, tmp_path: Path):
        p = make_multipart_eml(tmp_path, body="Multipart content")
        loader = EmlLoader()
        text = loader.load_as_text(str(p))
        assert "Multipart content" in text


# ---------------------------------------------------------------------------
# load_as_chunks
# ---------------------------------------------------------------------------


class TestLoadAsChunks:
    def test_returns_list_of_strings(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path, body="Short body")
        loader = EmlLoader()
        fake_chunks = [{"text": "chunk1", "token_count": 5, "start_index": 0, "end_index": 5}]
        with patch.object(loader._chunker, "chunk_text", return_value=fake_chunks):
            chunks = loader.load_as_chunks(str(p))
        assert isinstance(chunks, list)
        assert chunks == ["chunk1"]

    def test_file_not_found_raises(self):
        loader = EmlLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_as_chunks("/nonexistent/file.eml")

    def test_non_eml_raises(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("not an eml")
        loader = EmlLoader()
        with pytest.raises(ValueError):
            loader.load_as_chunks(str(f))

    def test_path_object_input(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path)
        loader = EmlLoader()
        fake_chunks = [{"text": "c", "token_count": 1, "start_index": 0, "end_index": 1}]
        with patch.object(loader._chunker, "chunk_text", return_value=fake_chunks):
            chunks = loader.load_as_chunks(p)
        assert isinstance(chunks, list)

    def test_run_delegates_to_load_as_chunks(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path)
        loader = EmlLoader()
        with patch.object(loader, "load_as_chunks", return_value=["chunk1"]) as mock:
            result = loader.run(str(p))
        mock.assert_called_once_with(str(p))
        assert result == ["chunk1"]


# ---------------------------------------------------------------------------
# load_as_nodes
# ---------------------------------------------------------------------------


class TestLoadAsNodes:
    def test_returns_single_node(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path, body="Email body")
        loader = EmlLoader()
        nodes = loader.load_as_nodes(str(p))
        assert len(nodes) == 1
        assert isinstance(nodes[0], Node)
        assert "Email body" in nodes[0].content

    def test_node_position_is_zero(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path)
        loader = EmlLoader()
        nodes = loader.load_as_nodes(str(p))
        assert nodes[0].metadata.position == 0

    def test_default_source_id(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path)
        loader = EmlLoader()
        nodes = loader.load_as_nodes(str(p))
        assert nodes[0].metadata.source_file_uuid == str(p)

    def test_custom_source_id(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path)
        loader = EmlLoader()
        nodes = loader.load_as_nodes(str(p), source_id="custom_id")
        assert nodes[0].metadata.source_file_uuid == "custom_id"

    def test_custom_metadata(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path)
        loader = EmlLoader()
        nodes = loader.load_as_nodes(str(p), custom_metadata={"tag": "work"})
        assert nodes[0].metadata.custom["tag"] == "work"

    def test_path_object_input(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path)
        loader = EmlLoader()
        nodes = loader.load_as_nodes(p)
        assert len(nodes) == 1


# ---------------------------------------------------------------------------
# load_as_vsfile
# ---------------------------------------------------------------------------


class TestLoadAsVsfile:
    def test_basic_vsfile(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path)
        loader = EmlLoader()
        vsfile = loader.load_as_vsfile(str(p))
        assert isinstance(vsfile, VSFile)
        assert vsfile.processed is True
        assert len(vsfile.nodes) == 1

    def test_file_not_found_raises(self):
        loader = EmlLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_as_vsfile("/nonexistent.eml")

    def test_non_eml_raises(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("not eml")
        loader = EmlLoader()
        with pytest.raises(ValueError):
            loader.load_as_vsfile(str(f))

    def test_path_object_raises_for_missing_file(self):
        loader = EmlLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_as_vsfile(Path("/nonexistent.eml"))

    def test_vsfile_nodes_have_vsfile_uuid_as_source(self, tmp_path: Path):
        p = make_plain_text_eml(tmp_path)
        loader = EmlLoader()
        vsfile = loader.load_as_vsfile(str(p))
        assert vsfile.nodes[0].metadata.source_file_uuid == str(vsfile.uuid)


# ---------------------------------------------------------------------------
# _extract_headers
# ---------------------------------------------------------------------------


class TestExtractHeaders:
    def test_extracts_subject_from_to(self, tmp_path: Path):
        p = make_plain_text_eml(
            tmp_path,
            subject="Test Subject",
            from_addr="a@a.com",
            to_addr="b@b.com",
        )
        loader = EmlLoader()
        text = loader.load_as_text(str(p))
        assert "Subject: Test Subject" in text
        assert "From: a@a.com" in text
        assert "To: b@b.com" in text

    def test_cc_extracted_when_present(self, tmp_path: Path):
        boundary = "bound"
        content = (
            "From: a@a.com\r\n"
            "To: b@b.com\r\n"
            "Cc: c@c.com\r\n"
            "Subject: CC Test\r\n"
            "Date: Mon, 01 Jan 2024 00:00:00 +0000\r\n"
            "Content-Type: text/plain\r\n"
            "\r\n"
            "Body\r\n"
        )
        p = tmp_path / "cc.eml"
        p.write_bytes(content.encode("utf-8"))
        loader = EmlLoader()
        text = loader.load_as_text(str(p))
        assert "Cc: c@c.com" in text


# ---------------------------------------------------------------------------
# create_eml_loader factory
# ---------------------------------------------------------------------------


class TestCreateEmlLoader:
    def test_factory_defaults(self):
        loader = create_eml_loader()
        assert isinstance(loader, EmlLoader)
        assert loader.config['chunk_size'] == 3000
        assert loader.config['include_headers'] is True

    def test_factory_custom(self):
        loader = create_eml_loader(
            chunk_size=1000,
            include_headers=False,
            extract_html=False,
        )
        assert loader.config['chunk_size'] == 1000
        assert loader.config['include_headers'] is False
        assert loader.config['extract_html'] is False

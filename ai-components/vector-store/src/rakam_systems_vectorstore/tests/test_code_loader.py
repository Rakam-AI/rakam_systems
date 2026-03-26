"""Tests for CodeLoader."""
from pathlib import Path
from unittest.mock import patch

import pytest

from rakam_systems_vectorstore.components.loader.code_loader import (
    CodeLoader,
    create_code_loader,
)
from rakam_systems_vectorstore.core import Node, VSFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk_result(texts):
    """Build list of chunk dicts matching the format chunk_text returns."""
    return [
        {
            "text": t,
            "token_count": len(t),
            "start_index": 0,
            "end_index": len(t),
        }
        for t in texts
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def py_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.py"
    p.write_text(
        "def hello():\n"
        "    return 'world'\n"
        "\n"
        "class Foo:\n"
        "    pass\n"
    )
    return p


@pytest.fixture
def js_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.js"
    p.write_text(
        "function greet(name) {\n"
        "    return `Hello ${name}`;\n"
        "}\n"
    )
    return p


@pytest.fixture
def go_file(tmp_path: Path) -> Path:
    p = tmp_path / "main.go"
    p.write_text(
        "package main\n\n"
        "func main() {\n"
        '    println("hello")\n'
        "}\n"
    )
    return p


@pytest.fixture
def loader() -> CodeLoader:
    return CodeLoader()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestCodeLoaderInit:
    def test_defaults(self):
        loader = CodeLoader()
        assert loader.config['chunk_size'] == 2000
        assert loader.config['chunk_overlap'] == 200
        assert loader.config['min_sentences_per_chunk'] == 3
        assert loader.config['tokenizer'] == "character"
        assert loader.config['preserve_structure'] is True
        assert loader.config['include_comments'] is True
        assert loader.config['encoding'] == "utf-8"

    def test_custom_config(self):
        loader = CodeLoader(
            config={
                "chunk_size": 500,
                "chunk_overlap": 50,
                "preserve_structure": False,
                "include_comments": False,
                "encoding": "latin-1",
            }
        )
        assert loader.config['chunk_size'] == 500
        assert loader.config['preserve_structure'] is False
        assert loader.config['include_comments'] is False
        assert loader.config['encoding'] == "latin-1"

    def test_custom_name(self):
        loader = CodeLoader(name="my_code_loader")
        assert loader.name == "my_code_loader"


# ---------------------------------------------------------------------------
# _is_code_file
# ---------------------------------------------------------------------------


class TestIsCodeFile:
    def test_python(self, loader):
        assert loader.is_code_file("script.py") is True

    def test_javascript(self, loader):
        assert loader.is_code_file("app.js") is True

    def test_typescript(self, loader):
        assert loader.is_code_file("app.ts") is True

    def test_go(self, loader):
        assert loader.is_code_file("main.go") is True

    def test_rust(self, loader):
        assert loader.is_code_file("lib.rs") is True

    def test_java(self, loader):
        assert loader.is_code_file("Main.java") is True

    def test_yaml(self, loader):
        assert loader.is_code_file("config.yaml") is True
        assert loader.is_code_file("config.yml") is True

    def test_json(self, loader):
        assert loader.is_code_file("data.json") is True

    def test_sql(self, loader):
        assert loader.is_code_file("query.sql") is True

    def test_unsupported_pdf(self, loader):
        assert loader.is_code_file("document.pdf") is False

    def test_unsupported_xlsx(self, loader):
        assert loader.is_code_file("data.xlsx") is False

    def test_unsupported_docx(self, loader):
        assert loader.is_code_file("doc.docx") is False

    def test_case_insensitive_py(self, loader):
        assert loader.is_code_file("script.PY") is True

    def test_case_insensitive_js(self, loader):
        assert loader.is_code_file("app.JS") is True


# ---------------------------------------------------------------------------
# _detect_language
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_python(self, loader):
        assert loader.detect_language("script.py") == "python"

    def test_javascript(self, loader):
        assert loader.detect_language("app.js") == "javascript"

    def test_typescript(self, loader):
        assert loader.detect_language("app.ts") == "typescript"

    def test_tsx(self, loader):
        assert loader.detect_language("comp.tsx") == "typescript"

    def test_jsx(self, loader):
        assert loader.detect_language("comp.jsx") == "javascript"

    def test_go(self, loader):
        assert loader.detect_language("main.go") == "go"

    def test_rust(self, loader):
        assert loader.detect_language("lib.rs") == "rust"

    def test_java(self, loader):
        assert loader.detect_language("Main.java") == "java"

    def test_cpp(self, loader):
        assert loader.detect_language("app.cpp") == "cpp"
        assert loader.detect_language("app.cc") == "cpp"

    def test_c(self, loader):
        assert loader.detect_language("app.c") == "c"

    def test_ruby(self, loader):
        assert loader.detect_language("app.rb") == "ruby"

    def test_shell(self, loader):
        assert loader.detect_language("script.sh") == "shell"

    def test_unknown_extension(self, loader):
        assert loader.detect_language("file.xyz") == "unknown"

    def test_case_insensitive(self, loader):
        assert loader.detect_language("script.PY") == "python"


# ---------------------------------------------------------------------------
# load_as_text
# ---------------------------------------------------------------------------


class TestLoadAsText:
    def test_python_file_content(self, loader, py_file: Path):
        text = loader.load_as_text(str(py_file))
        assert "def hello" in text
        assert "class Foo" in text

    def test_returns_full_content(self, loader, py_file: Path):
        original = py_file.read_text()
        text = loader.load_as_text(str(py_file))
        assert text == original

    def test_file_not_found_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_as_text("/nonexistent/file.py")

    def test_unsupported_extension_raises(self, loader, tmp_path: Path):
        f = tmp_path / "document.docx"
        f.write_bytes(b"fake docx")
        with pytest.raises(ValueError):
            loader.load_as_text(str(f))

    def test_path_object_input(self, loader, py_file: Path):
        text = loader.load_as_text(py_file)
        assert isinstance(text, str)

    def test_javascript_file(self, loader, js_file: Path):
        text = loader.load_as_text(str(js_file))
        assert "function greet" in text


# ---------------------------------------------------------------------------
# load_as_chunks
# ---------------------------------------------------------------------------


class TestLoadAsChunks:
    def test_with_mocked_structure_chunker(self, loader, py_file: Path):
        # Mock at the structure-chunking level since small files don't hit the text chunker
        with patch.object(
            loader, "_chunk_code_with_structure", return_value=["chunk1", "chunk2"]
        ):
            chunks = loader.load_as_chunks(str(py_file))
        assert chunks == ["chunk1", "chunk2"]

    def test_file_not_found_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_as_chunks("/nonexistent/file.py")

    def test_unsupported_extension_raises(self, loader, tmp_path: Path):
        f = tmp_path / "data.pdf"
        f.write_bytes(b"not code")
        with pytest.raises(ValueError):
            loader.load_as_chunks(str(f))

    def test_path_object_input(self, loader, py_file: Path):
        fake_chunks = make_chunk_result(["c"])
        with patch.object(loader._chunker, "chunk_text", return_value=fake_chunks):
            chunks = loader.load_as_chunks(py_file)
        assert isinstance(chunks, list)

    def test_no_preserve_structure_uses_text_chunker(self, py_file: Path):
        loader = CodeLoader(config={"preserve_structure": False})
        # When preserve_structure=False, _chunk_text is called which calls chunker
        fake_chunks = make_chunk_result(["text_chunk"])
        with patch.object(loader._chunker, "chunk_text", return_value=fake_chunks):
            chunks = loader.load_as_chunks(str(py_file))
        # The result is extracted text strings (not chunk dicts)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_run_delegates_to_load_as_chunks(self, loader, py_file: Path):
        with patch.object(loader, "load_as_chunks", return_value=["c1"]) as mock:
            result = loader.run(str(py_file))
        mock.assert_called_once_with(str(py_file))
        assert result == ["c1"]


# ---------------------------------------------------------------------------
# load_as_nodes
# ---------------------------------------------------------------------------


class TestLoadAsNodes:
    def test_node_count(self, loader, py_file: Path):
        with patch.object(
            loader, "_chunk_code_with_structure", return_value=["chunk_a", "chunk_b"]
        ):
            nodes = loader.load_as_nodes(str(py_file))
        assert len(nodes) == 2
        assert all(isinstance(n, Node) for n in nodes)

    def test_node_positions(self, loader, py_file: Path):
        with patch.object(
            loader, "_chunk_code_with_structure", return_value=["a", "b", "c"]
        ):
            nodes = loader.load_as_nodes(str(py_file))
        assert nodes[0].metadata.position == 0
        assert nodes[1].metadata.position == 1
        assert nodes[2].metadata.position == 2

    def test_node_has_language_metadata(self, loader, py_file: Path):
        with patch.object(
            loader, "_chunk_code_with_structure", return_value=["chunk"]
        ):
            nodes = loader.load_as_nodes(str(py_file))
        assert nodes[0].metadata.custom["language"] == "python"
        assert nodes[0].metadata.custom["file_extension"] == ".py"

    def test_custom_metadata_included(self, loader, py_file: Path):
        with patch.object(
            loader, "_chunk_code_with_structure", return_value=["chunk"]
        ):
            nodes = loader.load_as_nodes(
                str(py_file), custom_metadata={"project": "myapp"}
            )
        assert nodes[0].metadata.custom["project"] == "myapp"

    def test_default_source_id(self, loader, py_file: Path):
        with patch.object(
            loader, "_chunk_code_with_structure", return_value=["chunk"]
        ):
            nodes = loader.load_as_nodes(str(py_file))
        assert nodes[0].metadata.source_file_uuid == str(py_file)

    def test_custom_source_id(self, loader, py_file: Path):
        with patch.object(
            loader, "_chunk_code_with_structure", return_value=["chunk"]
        ):
            nodes = loader.load_as_nodes(str(py_file), source_id="my_id")
        assert nodes[0].metadata.source_file_uuid == "my_id"

    def test_javascript_node_language(self, loader, js_file: Path):
        with patch.object(
            loader, "_chunk_code_with_structure", return_value=["chunk"]
        ):
            nodes = loader.load_as_nodes(str(js_file))
        assert nodes[0].metadata.custom["language"] == "javascript"
        assert nodes[0].metadata.custom["file_extension"] == ".js"


# ---------------------------------------------------------------------------
# load_as_vsfile
# ---------------------------------------------------------------------------


class TestLoadAsVsfile:
    def test_basic_vsfile(self, loader, py_file: Path):
        fake_chunks = make_chunk_result(["chunk1"])
        with patch.object(loader._chunker, "chunk_text", return_value=fake_chunks):
            vsfile = loader.load_as_vsfile(str(py_file))
        assert isinstance(vsfile, VSFile)
        assert vsfile.processed is True
        assert len(vsfile.nodes) == 1

    def test_vsfile_multiple_nodes(self, loader, py_file: Path):
        with patch.object(
            loader, "_chunk_code_with_structure", return_value=["c1", "c2", "c3"]
        ):
            vsfile = loader.load_as_vsfile(str(py_file))
        assert len(vsfile.nodes) == 3

    def test_file_not_found_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_as_vsfile("/nonexistent.py")

    def test_unsupported_extension_raises(self, loader, tmp_path: Path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError):
            loader.load_as_vsfile(str(f))


# ---------------------------------------------------------------------------
# _split_by_blank_lines
# ---------------------------------------------------------------------------


class TestSplitByBlankLines:
    def test_splits_on_blank_lines(self, loader):
        content = "block1 line1\nblock1 line2\n\nblock2 line1\n\nblock3"
        blocks = loader._split_by_blank_lines(content)
        assert len(blocks) == 3

    def test_strips_blocks(self, loader):
        content = "  block1  \n\n  block2  "
        blocks = loader._split_by_blank_lines(content)
        assert blocks[0] == "block1"
        assert blocks[1] == "block2"

    def test_empty_content(self, loader):
        blocks = loader._split_by_blank_lines("")
        assert blocks == []

    def test_single_block(self, loader):
        content = "only one block\nwith multiple lines"
        blocks = loader._split_by_blank_lines(content)
        assert len(blocks) == 1


# ---------------------------------------------------------------------------
# _split_by_structure
# ---------------------------------------------------------------------------


class TestSplitByStructure:
    def test_python_splits_on_def(self, loader):
        content = (
            "import os\n"
            "\n"
            "def foo():\n"
            "    pass\n"
            "\n"
            "def bar():\n"
            "    pass\n"
        )
        blocks = loader._split_by_structure(content, "python")
        # Should have at least the 2 function blocks
        assert len(blocks) >= 2

    def test_python_splits_on_class(self, loader):
        content = (
            "class Foo:\n"
            "    pass\n"
            "\n"
            "class Bar:\n"
            "    pass\n"
        )
        blocks = loader._split_by_structure(content, "python")
        assert len(blocks) >= 2

    def test_unknown_language_falls_back_to_blank_lines(self, loader):
        content = "block1\n\nblock2"
        blocks = loader._split_by_structure(content, "unknown_lang_xyz")
        assert len(blocks) == 2

    def test_go_splits_on_func(self, loader):
        content = (
            "package main\n"
            "\n"
            "func main() {\n"
            "}\n"
            "\n"
            "func helper() {\n"
            "}\n"
        )
        blocks = loader._split_by_structure(content, "go")
        assert len(blocks) >= 2


# ---------------------------------------------------------------------------
# _get_structure_patterns
# ---------------------------------------------------------------------------


def test_python_patterns_not_empty(loader):
    patterns = loader._get_structure_patterns("python")
    assert len(patterns) > 0


def test_javascript_patterns_not_empty(loader):
    patterns = loader._get_structure_patterns("javascript")
    assert len(patterns) > 0


def test_unknown_language_returns_empty(loader):
    patterns = loader._get_structure_patterns("cobol")
    assert patterns == []


# ---------------------------------------------------------------------------
# _chunk_code_with_structure
# ---------------------------------------------------------------------------


class TestChunkCodeWithStructure:
    def test_empty_content_returns_empty_list(self, loader):
        chunks = loader._chunk_code_with_structure("", "python")
        assert chunks == []

    def test_whitespace_only_returns_empty(self, loader):
        chunks = loader._chunk_code_with_structure("   \n  ", "python")
        assert chunks == []

    def test_returns_list(self, loader, py_file: Path):
        content = py_file.read_text()
        chunks = loader._chunk_code_with_structure(content, "python")
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_content_preserved_in_chunks(self, loader):
        content = "def hello():\n    return 'world'\n"
        chunks = loader._chunk_code_with_structure(content, "python")
        combined = "\n\n".join(chunks)
        assert "hello" in combined


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_with_mocked_chunker(loader):
    fake_chunks = make_chunk_result(["part1", "part2"])
    with patch.object(loader._chunker, "chunk_text", return_value=fake_chunks):
        result = loader._chunk_text("some text", "python")
    assert result == ["part1", "part2"]


def test_chunk_text_empty_returns_empty(loader):
    result = loader._chunk_text("", "python")
    assert result == []


def test_chunk_text_falls_back_on_exception(loader):
    with patch.object(
        loader._chunker, "chunk_text", side_effect=RuntimeError("chunker error")
    ):
        result = loader._chunk_text("some text", "python")
    # Falls back to returning whole text as single chunk
    assert result == ["some text"]


# ---------------------------------------------------------------------------
# create_code_loader factory
# ---------------------------------------------------------------------------


class TestCreateCodeLoader:
    def test_factory_defaults(self):
        loader = create_code_loader()
        assert isinstance(loader, CodeLoader)
        assert loader.config['chunk_size'] == 2000
        assert loader.config['preserve_structure'] is True

    def test_factory_custom(self):
        loader = create_code_loader(
            chunk_size=500,
            chunk_overlap=50,
            preserve_structure=False,
            include_comments=False,
            encoding="latin-1",
        )
        assert loader.config['chunk_size'] == 500
        assert loader.config['chunk_overlap'] == 50
        assert loader.config['preserve_structure'] is False
        assert loader.config['include_comments'] is False
        assert loader.config['encoding'] == "latin-1"

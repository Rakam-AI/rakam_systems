"""Tests for TabularLoader."""
import csv
from pathlib import Path

import pytest

from rakam_systems_vectorstore.components.loader.tabular_loader import (
    CSV_EXTENSIONS,
    TABULAR_EXTENSIONS,
    TSV_EXTENSIONS,
    XLSX_EXTENSIONS,
    TabularLoader,
    create_tabular_loader,
)
from rakam_systems_vectorstore.core import Node, VSFile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.csv"
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "age", "city"])
        writer.writerow(["Alice", "30", "NYC"])
        writer.writerow(["Bob", "25", "LA"])
    return p


@pytest.fixture
def tsv_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.tsv"
    with p.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "value"])
        writer.writerow(["1", "foo"])
        writer.writerow(["2", "bar"])
    return p


@pytest.fixture
def csv_with_empty_rows(tmp_path: Path) -> Path:
    p = tmp_path / "empty_rows.csv"
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["col1", "col2"])
        writer.writerow(["a", "b"])
        writer.writerow(["", ""])  # empty row
        writer.writerow(["c", "d"])
    return p


@pytest.fixture
def no_header_csv(tmp_path: Path) -> Path:
    p = tmp_path / "no_header.csv"
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["val1", "val2"])
        writer.writerow(["val3", "val4"])
    return p


@pytest.fixture
def single_row_csv(tmp_path: Path) -> Path:
    """CSV with only header, no data."""
    p = tmp_path / "header_only.csv"
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["col1", "col2"])
    return p


# ---------------------------------------------------------------------------
# TabularLoader initialization
# ---------------------------------------------------------------------------


class TestTabularLoaderInit:
    def test_defaults(self):
        loader = TabularLoader()
        assert loader._sheet_name == 0
        assert loader._skip_empty_rows is True
        assert loader._empty_value_text == ""
        assert loader._delimiter is None
        assert loader._encoding == "utf-8"
        assert loader._has_header is True

    def test_custom_config(self):
        loader = TabularLoader(
            config={
                "sheet_name": "Sheet2",
                "skip_empty_rows": False,
                "empty_value_text": "<empty>",
                "delimiter": "|",
                "has_header": False,
            }
        )
        assert loader._sheet_name == "Sheet2"
        assert loader._skip_empty_rows is False
        assert loader._empty_value_text == "<empty>"
        assert loader._delimiter == "|"
        assert loader._has_header is False


# ---------------------------------------------------------------------------
# load_as_text
# ---------------------------------------------------------------------------


class TestLoadAsText:
    def test_csv_returns_string(self, csv_file: Path):
        loader = TabularLoader()
        text = loader.load_as_text(str(csv_file))
        assert isinstance(text, str)
        assert "Alice" in text
        assert "Bob" in text

    def test_csv_text_contains_all_fields(self, csv_file: Path):
        loader = TabularLoader()
        text = loader.load_as_text(str(csv_file))
        assert "name: Alice" in text
        assert "age: 30" in text
        assert "city: NYC" in text

    def test_file_not_found_raises(self):
        loader = TabularLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_as_text("/nonexistent/file.csv")

    def test_unsupported_extension_raises(self, tmp_path: Path):
        f = tmp_path / "data.pdf"
        f.write_text("not a csv")
        loader = TabularLoader()
        with pytest.raises(ValueError):
            loader.load_as_text(str(f))

    def test_path_object_input(self, csv_file: Path):
        loader = TabularLoader()
        text = loader.load_as_text(csv_file)
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# load_as_chunks
# ---------------------------------------------------------------------------


class TestLoadAsChunks:
    def test_csv_row_count(self, csv_file: Path):
        loader = TabularLoader()
        chunks = loader.load_as_chunks(str(csv_file))
        assert len(chunks) == 2  # Alice and Bob (header excluded)

    def test_chunk_format(self, csv_file: Path):
        loader = TabularLoader()
        chunks = loader.load_as_chunks(str(csv_file))
        assert "name: Alice" in chunks[0]
        assert "age: 30" in chunks[0]
        assert "city: NYC" in chunks[0]

    def test_tsv_loading(self, tsv_file: Path):
        loader = TabularLoader()
        chunks = loader.load_as_chunks(str(tsv_file))
        assert len(chunks) == 2
        assert "id: 1" in chunks[0]
        assert "value: foo" in chunks[0]

    def test_skip_empty_rows_enabled(self, csv_with_empty_rows: Path):
        loader = TabularLoader()
        chunks = loader.load_as_chunks(str(csv_with_empty_rows))
        assert len(chunks) == 2  # Only "a,b" and "c,d"

    def test_skip_empty_rows_disabled(self, csv_with_empty_rows: Path):
        loader = TabularLoader(config={"skip_empty_rows": False})
        chunks = loader.load_as_chunks(str(csv_with_empty_rows))
        assert len(chunks) == 3  # Includes empty row

    def test_path_object_input(self, csv_file: Path):
        loader = TabularLoader()
        chunks = loader.load_as_chunks(csv_file)
        assert len(chunks) == 2

    def test_run_delegates_to_load_as_chunks(self, csv_file: Path):
        loader = TabularLoader()
        chunks = loader.run(str(csv_file))
        assert isinstance(chunks, list)
        assert len(chunks) == 2

    def test_no_header_mode(self, no_header_csv: Path):
        loader = TabularLoader(config={"has_header": False})
        chunks = loader.load_as_chunks(str(no_header_csv))
        assert len(chunks) == 2
        assert "Column_1" in chunks[0]

    def test_header_only_csv_returns_empty(self, single_row_csv: Path):
        loader = TabularLoader()
        chunks = loader.load_as_chunks(str(single_row_csv))
        assert chunks == []

    def test_custom_delimiter(self, tmp_path: Path):
        p = tmp_path / "pipe.csv"
        p.write_text("col1|col2\nval1|val2\n")
        loader = TabularLoader(config={"delimiter": "|"})
        chunks = loader.load_as_chunks(str(p))
        assert len(chunks) == 1
        assert "col1: val1" in chunks[0]

    def test_file_not_found_raises(self):
        loader = TabularLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_as_chunks("/nonexistent/file.csv")

    def test_unsupported_extension_raises(self, tmp_path: Path):
        f = tmp_path / "data.pdf"
        f.write_text("not a csv")
        loader = TabularLoader()
        with pytest.raises(ValueError):
            loader.load_as_chunks(str(f))


# ---------------------------------------------------------------------------
# load_as_nodes
# ---------------------------------------------------------------------------


class TestLoadAsNodes:
    def test_csv_node_count(self, csv_file: Path):
        loader = TabularLoader()
        nodes = loader.load_as_nodes(str(csv_file))
        assert len(nodes) == 2
        assert all(isinstance(n, Node) for n in nodes)

    def test_node_positions(self, csv_file: Path):
        loader = TabularLoader()
        nodes = loader.load_as_nodes(str(csv_file))
        assert nodes[0].metadata.position == 0
        assert nodes[1].metadata.position == 1

    def test_default_source_id(self, csv_file: Path):
        loader = TabularLoader()
        nodes = loader.load_as_nodes(str(csv_file))
        assert nodes[0].metadata.source_file_uuid == str(csv_file)

    def test_custom_source_id(self, csv_file: Path):
        loader = TabularLoader()
        nodes = loader.load_as_nodes(str(csv_file), source_id="my_source")
        assert nodes[0].metadata.source_file_uuid == "my_source"

    def test_custom_metadata(self, csv_file: Path):
        loader = TabularLoader()
        nodes = loader.load_as_nodes(
            str(csv_file), custom_metadata={"source": "test", "version": 1}
        )
        assert nodes[0].metadata.custom["source"] == "test"
        assert nodes[0].metadata.custom["version"] == 1

    def test_node_content(self, csv_file: Path):
        loader = TabularLoader()
        nodes = loader.load_as_nodes(str(csv_file))
        assert "name: Alice" in nodes[0].content


# ---------------------------------------------------------------------------
# load_as_vsfile
# ---------------------------------------------------------------------------


class TestLoadAsVsfile:
    def test_csv_vsfile(self, csv_file: Path):
        loader = TabularLoader()
        vsfile = loader.load_as_vsfile(str(csv_file))
        assert isinstance(vsfile, VSFile)
        assert vsfile.processed is True
        assert len(vsfile.nodes) == 2

    def test_vsfile_nodes_have_correct_source_id(self, csv_file: Path):
        loader = TabularLoader()
        vsfile = loader.load_as_vsfile(str(csv_file))
        assert vsfile.nodes[0].metadata.source_file_uuid == str(vsfile.uuid)

    def test_file_not_found(self):
        loader = TabularLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_as_vsfile("/nonexistent.csv")

    def test_unsupported_file_raises(self, tmp_path: Path):
        f = tmp_path / "bad.txt"
        f.write_text("bad")
        loader = TabularLoader()
        with pytest.raises(ValueError):
            loader.load_as_vsfile(str(f))

    def test_path_object_raises_file_not_found(self):
        loader = TabularLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_as_vsfile(Path("/nonexistent.csv"))


# ---------------------------------------------------------------------------
# _format_row
# ---------------------------------------------------------------------------


class TestFormatRow:
    def test_basic_format(self):
        loader = TabularLoader()
        result = loader._format_row(["name", "age"], ("Alice", "30"))
        assert "name: Alice" in result
        assert "age: 30" in result

    def test_fields_separated_by_double_newline(self):
        loader = TabularLoader()
        result = loader._format_row(["a", "b"], ("x", "y"))
        assert "\n\n" in result

    def test_none_value_shows_empty(self):
        loader = TabularLoader()
        result = loader._format_row(["col"], (None,))
        assert "col: " in result

    def test_empty_string_value_shows_empty(self):
        loader = TabularLoader()
        result = loader._format_row(["col"], ("",))
        assert "col: " in result

    def test_custom_empty_value_text(self):
        loader = TabularLoader(config={"empty_value_text": "<empty>"})
        result = loader._format_row(["col1", "col2"], ("value", None))
        assert "col2: <empty>" in result

    def test_numeric_value_converted_to_string(self):
        loader = TabularLoader()
        result = loader._format_row(["score"], (99.5,))
        assert "score: 99.5" in result


# ---------------------------------------------------------------------------
# _get_file_type
# ---------------------------------------------------------------------------


class TestGetFileType:
    def test_csv(self):
        loader = TabularLoader()
        assert loader._get_file_type("data.csv") == "csv"

    def test_tsv(self):
        loader = TabularLoader()
        assert loader._get_file_type("data.tsv") == "tsv"

    def test_xlsx(self):
        loader = TabularLoader()
        assert loader._get_file_type("data.xlsx") == "xlsx"

    def test_unknown(self):
        loader = TabularLoader()
        assert loader._get_file_type("data.pdf") == "unknown"


# ---------------------------------------------------------------------------
# _is_supported_file
# ---------------------------------------------------------------------------


def test_is_supported_file_csv():
    loader = TabularLoader()
    assert loader._is_supported_file("data.csv") is True


def test_is_supported_file_tsv():
    loader = TabularLoader()
    assert loader._is_supported_file("data.tsv") is True


def test_is_supported_file_xlsx():
    loader = TabularLoader()
    assert loader._is_supported_file("data.xlsx") is True


def test_is_not_supported_file():
    loader = TabularLoader()
    assert loader._is_supported_file("data.pdf") is False
    assert loader._is_supported_file("data.docx") is False


# ---------------------------------------------------------------------------
# create_tabular_loader factory
# ---------------------------------------------------------------------------


class TestCreateTabularLoader:
    def test_factory_defaults(self):
        loader = create_tabular_loader()
        assert isinstance(loader, TabularLoader)
        assert loader._has_header is True

    def test_factory_custom(self):
        loader = create_tabular_loader(
            sheet_name="Sheet1",
            skip_empty_rows=False,
            empty_value_text="N/A",
            delimiter="|",
            has_header=False,
        )
        assert loader._sheet_name == "Sheet1"
        assert loader._skip_empty_rows is False
        assert loader._empty_value_text == "N/A"
        assert loader._delimiter == "|"
        assert loader._has_header is False


# ---------------------------------------------------------------------------
# Extension constants
# ---------------------------------------------------------------------------


def test_csv_in_extensions():
    assert ".csv" in CSV_EXTENSIONS
    assert ".csv" in TABULAR_EXTENSIONS


def test_tsv_in_extensions():
    assert ".tsv" in TSV_EXTENSIONS
    assert ".tsv" in TABULAR_EXTENSIONS


def test_xlsx_in_extensions():
    assert ".xlsx" in XLSX_EXTENSIONS
    assert ".xlsx" in TABULAR_EXTENSIONS

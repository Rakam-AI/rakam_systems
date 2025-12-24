"""
Tabular Data Loader for loading spreadsheet and tabular files without chunking.

This loader extracts data from tabular files (XLSX, CSV, TSV, etc.) where each row 
becomes a single node. Each node's content is formatted with each field on its own 
line, separated by blank lines:

    Column_name_1: value_1
    
    Column_name_2: value_2
    
    ...

Empty values are explicitly shown in the node content.

Features:
- No chunking - each row is one node
- All columns included with their names
- Readable format with blank line separation between fields
- Empty values explicitly shown
- Supports Excel files (.xlsx, .xls)
- Supports CSV files (.csv)
- Supports TSV files (.tsv)
- Supports other delimiter-separated files
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rakam_systems.core.ai_utils import logging
from rakam_systems.core.ai_core.interfaces.loader import Loader
from rakam_systems.vectorestore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)

# Supported file extensions and their types
XLSX_EXTENSIONS = ['.xlsx', '.xls']
CSV_EXTENSIONS = ['.csv']
TSV_EXTENSIONS = ['.tsv']
TABULAR_EXTENSIONS = XLSX_EXTENSIONS + CSV_EXTENSIONS + TSV_EXTENSIONS


class TabularLoader(Loader):
    """
    Tabular data loader that converts each row into a node without chunking.

    This loader provides simple tabular file processing where:
    - Each row becomes one node (chunk)
    - First row is treated as headers (column names)
    - Node content format: each field on its own line, separated by blank lines
    - Empty values are explicitly shown as empty
    - No chunking is performed

    Supported file types:
    - Excel files (.xlsx, .xls)
    - CSV files (.csv)
    - TSV files (.tsv)
    - Other delimiter-separated files (configurable delimiter)
    """

    def __init__(
        self,
        name: str = "tabular_loader",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize tabular data loader.

        Args:
            name: Component name
            config: Optional configuration with keys:
                - sheet_name: Name or index of sheet to load for Excel files (default: 0 for first sheet)
                - skip_empty_rows: Whether to skip completely empty rows (default: True)
                - empty_value_text: Text to show for empty values (default: "")
                - delimiter: Delimiter for CSV/text files (default: auto-detect based on extension)
                - encoding: File encoding for CSV/text files (default: 'utf-8')
                - has_header: Whether first row contains headers (default: True)
        """
        super().__init__(name=name, config=config)

        # Extract configuration
        config = config or {}
        self._sheet_name = config.get('sheet_name', 0)  # 0 = first sheet
        self._skip_empty_rows = config.get('skip_empty_rows', True)
        self._empty_value_text = config.get('empty_value_text', "")
        self._delimiter = config.get('delimiter', None)  # None = auto-detect
        self._encoding = config.get('encoding', 'utf-8')
        self._has_header = config.get('has_header', True)

        logger.info(
            f"Initialized TabularLoader with sheet_name={self._sheet_name}, skip_empty_rows={self._skip_empty_rows}")

    def run(self, source: str) -> List[str]:
        """
        Execute the primary operation for the component.

        This method satisfies the BaseComponent abstract method requirement
        and delegates to load_as_chunks.

        Args:
            source: Path to tabular file (XLSX, CSV, TSV, etc.)

        Returns:
            List of formatted strings, one per row (excluding header)
        """
        return self.load_as_chunks(source)

    def load_as_text(
        self,
        source: Union[str, Path],
    ) -> str:
        """
        Load tabular file and return as a single text string.

        This method extracts all rows from the file and returns them as a single
        string without chunking. Each row is on its own line.

        Args:
            source: Path to tabular file (XLSX, CSV, TSV, etc.)

        Returns:
            Full text content as a single string (all rows joined by newlines)

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not a supported tabular file
            Exception: If file processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file type
        if not self._is_supported_file(source):
            raise ValueError(
                f"File is not a supported tabular format: {source}")

        logger.info(f"Loading tabular file as text: {source}")
        start_time = time.time()

        try:
            # Get all rows
            row_strings = self._extract_rows(source)

            # Join all rows into a single string
            full_text = "\n".join(row_strings)

            elapsed = time.time() - start_time
            logger.info(
                f"Tabular file loaded as text in {elapsed:.2f}s: {len(full_text)} characters, {len(row_strings)} rows")

            return full_text

        except Exception as e:
            logger.error(f"Error loading tabular file as text {source}: {e}")
            raise

    def load_as_chunks(
        self,
        source: Union[str, Path],
    ) -> List[str]:
        """
        Load tabular file and return as a list of text chunks.

        Each row becomes one chunk (no text-based chunking is performed).
        Each chunk is formatted with each field on its own line, separated by blank lines.

        Args:
            source: Path to tabular file (XLSX, CSV, TSV, etc.)

        Returns:
            List of formatted strings, one per row (excluding header)

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not a supported tabular file
            Exception: If file processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file type
        if not self._is_supported_file(source):
            raise ValueError(
                f"File is not a supported tabular format: {source}")

        logger.info(f"Loading tabular file: {source}")
        start_time = time.time()

        try:
            row_strings = self._extract_rows(source)

            elapsed = time.time() - start_time
            logger.info(
                f"Tabular file processed in {elapsed:.2f}s: {len(row_strings)} rows")

            return row_strings

        except Exception as e:
            logger.error(f"Error processing tabular file {source}: {e}")
            raise

    def _get_file_type(self, source: str) -> str:
        """
        Determine the file type based on extension.

        Args:
            source: Path to file

        Returns:
            File type: 'xlsx', 'csv', 'tsv', or 'unknown'
        """
        ext = Path(source).suffix.lower()
        if ext in XLSX_EXTENSIONS:
            return 'xlsx'
        elif ext in CSV_EXTENSIONS:
            return 'csv'
        elif ext in TSV_EXTENSIONS:
            return 'tsv'
        return 'unknown'

    def _extract_rows(self, source: str) -> List[str]:
        """
        Extract rows from tabular file and format them as strings.

        Args:
            source: Path to tabular file

        Returns:
            List of formatted row strings
        """
        file_type = self._get_file_type(source)

        if file_type == 'xlsx':
            return self._extract_rows_xlsx(source)
        elif file_type in ('csv', 'tsv'):
            return self._extract_rows_csv(source, file_type)
        else:
            # Try to load as CSV with auto-detected or configured delimiter
            return self._extract_rows_csv(source, 'csv')

    def _extract_rows_xlsx(self, source: str) -> List[str]:
        """
        Extract rows from XLSX file and format them as strings.

        Args:
            source: Path to XLSX file

        Returns:
            List of formatted row strings
        """
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "openpyxl is required for XLSX loading. Install it with: pip install openpyxl"
            )

        # Load workbook
        workbook = openpyxl.load_workbook(source, data_only=True)

        # Get the specified sheet
        if isinstance(self._sheet_name, int):
            sheet = workbook.worksheets[self._sheet_name]
        else:
            sheet = workbook[self._sheet_name]

        logger.info(f"Processing sheet: {sheet.title}")

        # Extract rows
        rows = list(sheet.iter_rows(values_only=True))

        if not rows:
            logger.warning(f"No rows found in sheet {sheet.title}")
            return []

        return self._process_raw_rows(rows)

    def _extract_rows_csv(self, source: str, file_type: str) -> List[str]:
        """
        Extract rows from CSV/TSV file and format them as strings.

        Args:
            source: Path to CSV/TSV file
            file_type: 'csv' or 'tsv'

        Returns:
            List of formatted row strings
        """
        # Determine delimiter
        if self._delimiter is not None:
            delimiter = self._delimiter
        elif file_type == 'tsv':
            delimiter = '\t'
        else:
            delimiter = ','

        logger.info(
            f"Processing {file_type.upper()} file with delimiter: {repr(delimiter)}")

        rows = []
        with open(source, 'r', encoding=self._encoding, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                rows.append(tuple(row))

        if not rows:
            logger.warning(f"No rows found in file: {source}")
            return []

        return self._process_raw_rows(rows)

    def _process_raw_rows(self, rows: List[tuple]) -> List[str]:
        """
        Process raw rows (from any source) into formatted strings.

        Args:
            rows: List of row tuples

        Returns:
            List of formatted row strings
        """
        if not rows:
            return []

        # First row is headers (if configured)
        if self._has_header:
            headers = rows[0]
            data_rows = rows[1:]
            start_row_idx = 2  # For logging purposes
        else:
            # Generate column names if no header
            headers = tuple(f"Column_{i+1}" for i in range(len(rows[0])))
            data_rows = rows
            start_row_idx = 1

        # Convert None/empty headers to column names
        headers = [str(h) if h is not None and str(h).strip() != "" else f"Column_{i+1}"
                   for i, h in enumerate(headers)]

        logger.info(f"Found {len(headers)} columns: {headers}")

        # Process data rows
        row_strings = []
        for row_idx, row in enumerate(data_rows, start=start_row_idx):
            # Skip empty rows if configured
            if self._skip_empty_rows and all(
                cell is None or str(cell).strip() == "" for cell in row
            ):
                logger.debug(f"Skipping empty row {row_idx}")
                continue

            # Format row
            row_string = self._format_row(headers, row)
            row_strings.append(row_string)

        return row_strings

    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        Load tabular file and return as Node objects with metadata.

        Args:
            source: Path to tabular file (XLSX, CSV, TSV, etc.)
            source_id: Optional source identifier (defaults to file path)
            custom_metadata: Optional custom metadata to attach to nodes

        Returns:
            List of Node objects, one per row with metadata
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Load row strings
        row_strings = self.load_as_chunks(source)

        # Determine source ID
        if source_id is None:
            source_id = source

        # Create nodes with metadata
        nodes = []
        for idx, row_string in enumerate(row_strings):
            metadata = NodeMetadata(
                source_file_uuid=source_id,
                position=idx,
                custom=custom_metadata or {}
            )
            node = Node(content=row_string, metadata=metadata)
            nodes.append(node)

        logger.info(f"Created {len(nodes)} nodes from tabular file: {source}")
        return nodes

    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VSFile:
        """
        Load tabular file and return as VSFile object.

        Args:
            file_path: Path to tabular file (XLSX, CSV, TSV, etc.)
            custom_metadata: Optional custom metadata

        Returns:
            VSFile object with nodes

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a supported tabular format
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._is_supported_file(file_path):
            raise ValueError(
                f"File is not a supported tabular format: {file_path}")

        # Create VSFile
        vsfile = VSFile(file_path)

        # Load and create nodes
        nodes = self.load_as_nodes(
            file_path, str(vsfile.uuid), custom_metadata)
        vsfile.nodes = nodes
        vsfile.processed = True

        logger.info(
            f"Created VSFile with {len(nodes)} nodes from: {file_path}")
        return vsfile

    def _is_supported_file(self, file_path: str) -> bool:
        """
        Check if file is a supported tabular format based on extension.

        Args:
            file_path: Path to file

        Returns:
            True if file is a supported tabular format, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in TABULAR_EXTENSIONS

    def _is_xlsx_file(self, file_path: str) -> bool:
        """
        Check if file is an XLSX based on extension.

        Args:
            file_path: Path to file

        Returns:
            True if file is an XLSX, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in XLSX_EXTENSIONS

    def _format_row(self, headers: List[str], row: tuple) -> str:
        """
        Format a row with each field on its own line, separated by blank lines.

        Format:
            Column_name_1: value_1

            Column_name_2: value_2

            ...

        Args:
            headers: List of column names
            row: Tuple of cell values

        Returns:
            Formatted string representation of the row
        """
        parts = []

        for header, value in zip(headers, row):
            # Convert value to string, handling None and empty values
            if value is None or (isinstance(value, str) and value.strip() == ""):
                value_str = self._empty_value_text
            else:
                value_str = str(value)

            # Add to parts
            parts.append(f"{header}: {value_str}")

        # Join with double newline for better readability
        return "\n\n".join(parts)


def create_tabular_loader(
    sheet_name: Union[int, str] = 0,
    skip_empty_rows: bool = True,
    empty_value_text: str = "",
    delimiter: Optional[str] = None,
    encoding: str = 'utf-8',
    has_header: bool = True
) -> TabularLoader:
    """
    Factory function to create a tabular data loader.

    Args:
        sheet_name: Name or index of sheet to load for Excel files (default: 0 for first sheet)
        skip_empty_rows: Whether to skip completely empty rows (default: True)
        empty_value_text: Text to show for empty values (default: "")
        delimiter: Delimiter for CSV/text files (default: auto-detect based on extension)
        encoding: File encoding for CSV/text files (default: 'utf-8')
        has_header: Whether first row contains headers (default: True)

    Returns:
        Configured tabular loader

    Example:
        >>> # Load Excel file
        >>> loader = create_tabular_loader()
        >>> rows = loader.run("data/spreadsheet.xlsx")
        >>> print(f"Extracted {len(rows)} rows")

        >>> # Load CSV file
        >>> loader = create_tabular_loader()
        >>> rows = loader.run("data/data.csv")

        >>> # Load TSV file
        >>> loader = create_tabular_loader()
        >>> rows = loader.run("data/data.tsv")

        >>> # Load file with custom delimiter (e.g., pipe-separated)
        >>> loader = create_tabular_loader(delimiter='|')
        >>> rows = loader.run("data/data.txt")

        >>> # Load specific sheet from Excel
        >>> loader = create_tabular_loader(sheet_name="Sheet2")
        >>> rows = loader.run("data/spreadsheet.xlsx")

        >>> # Load file without header row
        >>> loader = create_tabular_loader(has_header=False)
        >>> rows = loader.run("data/data.csv")

        >>> # Show empty values explicitly
        >>> loader = create_tabular_loader(empty_value_text="<empty>")
        >>> rows = loader.run("data/spreadsheet.xlsx")
    """
    config = {
        'sheet_name': sheet_name,
        'skip_empty_rows': skip_empty_rows,
        'empty_value_text': empty_value_text,
        'delimiter': delimiter,
        'encoding': encoding,
        'has_header': has_header
    }

    return TabularLoader(config=config)


# Backward compatibility aliases
XlsxLoader = TabularLoader
create_xlsx_loader = create_tabular_loader


__all__ = [
    "TabularLoader",
    "create_tabular_loader",
    # Backward compatibility
    "XlsxLoader",
    "create_xlsx_loader",
    # Constants
    "TABULAR_EXTENSIONS",
    "XLSX_EXTENSIONS",
    "CSV_EXTENSIONS",
    "TSV_EXTENSIONS",
]

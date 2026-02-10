# Tabular Loader

A simple loader for tabular files (XLSX, CSV, TSV) that converts each row into a node without chunking.

## Features

- **No Chunking**: Each row becomes exactly one node
- **Column Names Preserved**: First row is treated as headers
- **Empty Values Shown**: Empty cells are explicitly shown in the output
- **Simple Format**: Each node's content is formatted as: `Column_name_1: value_1; Column_name_2: value_2; ...`
- **Multiple Formats**: Supports Excel (.xlsx, .xls), CSV (.csv), and TSV (.tsv) files

## Installation

For Excel file support, the `openpyxl` library is required:

```bash
pip install openpyxl
```

CSV and TSV support is built-in (no additional dependencies).

## Usage

### Basic Usage

```python
from rakam_system_vectorstore.components.loader.tabular_loader import create_tabular_loader

# Create loader
loader = create_tabular_loader()

# Load as strings (one per row)
rows = loader.run("data/spreadsheet.xlsx")  # Also works with .csv, .tsv
for row in rows:
    print(row)
```

### Load as Nodes

```python
# Load as Node objects with metadata
nodes = loader.load_as_nodes("data/spreadsheet.xlsx")

for node in nodes:
    print(f"Content: {node.content}")
    print(f"Metadata: {node.metadata}")
```

### Load as VSFile

```python
# Load as VSFile object
vsfile = loader.load_as_vsfile("data/spreadsheet.xlsx")

print(f"File: {vsfile.file_name}")
print(f"Nodes: {len(vsfile.nodes)}")
```

## Configuration Options

### Sheet Selection (Excel only)

Load a specific sheet by name or index:

```python
# Load first sheet (default)
loader = create_tabular_loader(sheet_name=0)

# Load specific sheet by name
loader = create_tabular_loader(sheet_name="Sheet2")

# Load second sheet by index
loader = create_tabular_loader(sheet_name=1)
```

### Custom Delimiter (CSV/TSV)

```python
# Auto-detected based on file extension (default)
loader = create_tabular_loader()

# Custom delimiter (e.g., pipe-separated)
loader = create_tabular_loader(delimiter='|')
```

### Empty Value Handling

Control how empty cells are displayed:

```python
# Default: empty string for empty cells
loader = create_tabular_loader(empty_value_text="")

# Show empty cells explicitly
loader = create_tabular_loader(empty_value_text="<empty>")

# Show as N/A
loader = create_tabular_loader(empty_value_text="N/A")
```

### Empty Row Handling

Control whether to skip completely empty rows:

```python
# Skip empty rows (default)
loader = create_tabular_loader(skip_empty_rows=True)

# Keep empty rows
loader = create_tabular_loader(skip_empty_rows=False)
```

### File Encoding (CSV/TSV)

```python
# Default encoding
loader = create_tabular_loader(encoding='utf-8')

# Different encoding
loader = create_tabular_loader(encoding='latin-1')
```

### Header Row

```python
# First row is header (default)
loader = create_tabular_loader(has_header=True)

# No header row - columns named Column_1, Column_2, etc.
loader = create_tabular_loader(has_header=False)
```

## Example

Given a tabular file with this content:

| Name    | Age | City       | Country |
|---------|-----|------------|---------|
| Alice   | 30  | New York   | USA     |
| Bob     |     | London     | UK      |
| Charlie | 25  |            | Canada  |

The loader will produce these nodes:

```
Row 1: Name: Alice; Age: 30; City: New York; Country: USA
Row 2: Name: Bob; Age: ; City: London; Country: UK
Row 3: Name: Charlie; Age: 25; City: ; Country: Canada
```

With `empty_value_text="<empty>"`:

```
Row 1: Name: Alice; Age: 30; City: New York; Country: USA
Row 2: Name: Bob; Age: <empty>; City: London; Country: UK
Row 3: Name: Charlie; Age: 25; City: <empty>; Country: Canada
```

## Testing

A test script is provided in `app/scripts/test_tabular_loader.py`:

```bash
python app/scripts/test_tabular_loader.py data/your_file.xlsx
python app/scripts/test_tabular_loader.py data/your_file.csv
```

## API Reference

### TabularLoader

Main loader class for tabular files.

**Constructor Parameters:**
- `name` (str): Component name (default: "tabular_loader")
- `config` (dict): Configuration options
  - `sheet_name` (int|str): Sheet to load for Excel files (default: 0)
  - `skip_empty_rows` (bool): Skip empty rows (default: True)
  - `empty_value_text` (str): Text for empty cells (default: "")
  - `delimiter` (str): Delimiter for CSV/TSV files (default: auto-detect)
  - `encoding` (str): File encoding for CSV/TSV (default: 'utf-8')
  - `has_header` (bool): Whether first row is header (default: True)

**Methods:**

#### `run(source: str) -> List[str]`
Load tabular file and return list of formatted row strings.

**Parameters:**
- `source`: Path to tabular file

**Returns:**
- List of strings, one per row (excluding header)

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file is not a supported tabular format

#### `load_as_text(source) -> str`
Load tabular file and return as a single text string.

**Parameters:**
- `source`: Path to tabular file

**Returns:**
- Full text content as a single string (all rows joined)

#### `load_as_chunks(source) -> List[str]`
Load tabular file and return as a list of text chunks (same as run()).

**Parameters:**
- `source`: Path to tabular file

**Returns:**
- List of strings, one per row

#### `load_as_nodes(source, source_id=None, custom_metadata=None) -> List[Node]`
Load tabular file and return as Node objects.

**Parameters:**
- `source`: Path to tabular file
- `source_id`: Optional source identifier
- `custom_metadata`: Optional custom metadata dict

**Returns:**
- List of Node objects with metadata

#### `load_as_vsfile(file_path, custom_metadata=None) -> VSFile`
Load tabular file and return as VSFile object.

**Parameters:**
- `file_path`: Path to tabular file
- `custom_metadata`: Optional custom metadata dict

**Returns:**
- VSFile object with nodes

### create_tabular_loader()

Factory function to create a tabular loader.

**Parameters:**
- `sheet_name` (int|str): Sheet to load for Excel files (default: 0)
- `skip_empty_rows` (bool): Skip empty rows (default: True)
- `empty_value_text` (str): Text for empty cells (default: "")
- `delimiter` (str): Delimiter for CSV/TSV (default: auto-detect)
- `encoding` (str): File encoding (default: 'utf-8')
- `has_header` (bool): First row is header (default: True)

**Returns:**
- Configured TabularLoader instance

## Backward Compatibility

For backward compatibility, `XlsxLoader` and `create_xlsx_loader` are available as aliases:

```python
from rakam_system_vectorstore.components.loader.tabular_loader import (
    XlsxLoader,  # Alias for TabularLoader
    create_xlsx_loader  # Alias for create_tabular_loader
)
```

## Notes

- The first row is always treated as headers (column names) by default
- If a header cell is empty, it will be named `Column_1`, `Column_2`, etc.
- Each row becomes exactly one node - no chunking is performed
- Empty cells are included in the output with configurable text
- Supports Excel (.xlsx, .xls), CSV (.csv), and TSV (.tsv) file extensions

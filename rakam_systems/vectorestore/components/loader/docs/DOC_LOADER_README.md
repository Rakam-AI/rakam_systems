# DOC/DOCX Loader

The `DocLoader` is a specialized loader for Microsoft Word documents (`.doc` and `.docx` files). It extracts text content, tables, and images from Word documents and provides them in various formats suitable for vector store indexing.

## Features

- **DOCX Support**: Full support for modern `.docx` format using `python-docx`
- **Legacy DOC Support**: Support for legacy `.doc` files via `antiword` (Linux/macOS) or `textutil` (macOS)
- **Text Extraction**: Extracts paragraphs and tables with proper formatting
- **Image Extraction**: Extracts embedded images from DOCX files to a scratch directory
- **Configurable Chunking**: Advanced text chunking with configurable size and overlap
- **Multiple Output Formats**: Returns text, chunks, nodes, or VSFile objects

## Installation

The DocLoader requires `python-docx` for DOCX processing:

```bash
pip install python-docx
```

For legacy `.doc` file support, you'll need one of:

- **antiword** (Linux/macOS): `apt install antiword` or `brew install antiword`
- **textutil** (macOS): Pre-installed on macOS

## Quick Start

### Basic Usage

```python
from rakam_systems.ai_vectorstore.components.loader import DocLoader, create_doc_loader

# Using factory function (recommended)
loader = create_doc_loader()

# Load as text chunks (default)
chunks = loader.run("document.docx")

# Load as full text
text = loader.load_as_text("document.docx")

# Load as nodes for vector store
nodes = loader.load_as_nodes("document.docx")

# Load as VSFile object
vsfile = loader.load_as_vsfile("document.docx")
```

### Custom Configuration

```python
loader = create_doc_loader(
    chunk_size=1024,           # Maximum tokens per chunk
    chunk_overlap=64,          # Overlap between chunks
    extract_tables=True,       # Include table content
    save_images=True,          # Extract images to scratch folder
    include_images_in_text=True,  # Add image references to text
    preserve_formatting=True   # Add heading markers (e.g., ## Heading)
)

chunks = loader.run("report.docx")
```

### Using DocLoader Class Directly

```python
from rakam_systems.ai_vectorstore.components.loader import DocLoader

loader = DocLoader(config={
    'chunk_size': 2048,
    'chunk_overlap': 128,
    'extract_tables': True,
    'save_images': True,
    'scratch_folder_name': 'extracted_images'
})

# Load with custom metadata
nodes = loader.load_as_nodes(
    "document.docx",
    source_id="custom_id",
    custom_metadata={"category": "reports", "author": "John Doe"}
)
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | int | 2048 | Maximum tokens per chunk |
| `chunk_overlap` | int | 128 | Overlap between consecutive chunks |
| `min_sentences_per_chunk` | int | 1 | Minimum sentences to include in each chunk |
| `tokenizer` | str | "character" | Tokenizer type: "character", "gpt2", or HuggingFace model ID |
| `embed_model_id` | str | "sentence-transformers/all-MiniLM-L6-v2" | Model ID for tokenization |
| `save_images` | bool | True | Whether to extract and save images |
| `scratch_folder_name` | str | "scratch" | Name of folder to store extracted images |
| `include_images_in_text` | bool | True | Whether to append image references to text |
| `extract_tables` | bool | True | Whether to include table content |
| `preserve_formatting` | bool | False | Whether to add markdown formatting markers |

## Output Formats

### `run(source)` / `load_as_chunks(source)`

Returns a list of text chunks:

```python
chunks = loader.run("document.docx")
# ['First chunk of text...', 'Second chunk of text...', ...]
```

### `load_as_text(source)`

Returns the full document text as a single string:

```python
text = loader.load_as_text("document.docx")
# 'Full document content including all paragraphs and tables...'
```

### `load_as_nodes(source, source_id, custom_metadata)`

Returns a list of `Node` objects with metadata:

```python
nodes = loader.load_as_nodes("document.docx")
for node in nodes:
    print(f"Content: {node.content[:50]}...")
    print(f"Position: {node.metadata.position}")
    print(f"Source: {node.metadata.source_file_uuid}")
```

### `load_as_vsfile(file_path, custom_metadata)`

Returns a `VSFile` object ready for vector store integration:

```python
vsfile = loader.load_as_vsfile("document.docx")
print(f"File UUID: {vsfile.uuid}")
print(f"Nodes: {len(vsfile.nodes)}")
print(f"Processed: {vsfile.processed}")
```

## Image Extraction

For DOCX files, images are extracted from the `word/media/` directory within the document archive. Supported image formats:

- PNG, JPG/JPEG, GIF, BMP, TIFF
- EMF, WMF (Windows Metafiles)

Images are saved to a scratch directory and optionally referenced in the extracted text:

```
--- Embedded Images ---

![image1.png](/path/to/scratch/image1.png)
![image2.jpg](/path/to/scratch/image2.jpg)
```

## Legacy .doc File Support

Legacy `.doc` files use the older binary format. The DocLoader attempts extraction via:

1. **python-docx**: Some .doc files are actually DOCX in disguise
2. **antiword**: Command-line tool for Linux/macOS
3. **textutil**: Built-in macOS utility

If all methods fail, an error is raised with installation suggestions.

## Integration with AdaptiveLoader

The `DocLoader` can also be used through the `AdaptiveLoader`, which automatically detects file types:

```python
from rakam_systems.ai_vectorstore.components.loader import AdaptiveLoader

loader = AdaptiveLoader()
chunks = loader.run("document.docx")  # Automatically uses DocLoader logic
```

## Error Handling

```python
from rakam_systems.ai_vectorstore.components.loader import DocLoader

loader = DocLoader()

try:
    chunks = loader.run("document.docx")
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Invalid file type: {e}")
except ImportError:
    print("python-docx not installed")
except RuntimeError as e:
    print(f"Failed to process .doc file: {e}")
```

## See Also

- [AdaptiveLoader](./adaptive_loader.py) - Automatic file type detection
- [OdtLoader](./ODT_LOADER_README.md) - OpenDocument text files
- [PdfLoader](./PDF_LOADER_ARCHITECTURE.md) - PDF documents

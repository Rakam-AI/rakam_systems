# EML Loader Documentation

## Overview

The `EmlLoader` is a specialized loader for processing email files in EML format (`.eml`). It extracts text content from email headers and body, supporting both plain text and HTML email formats. The loader uses `TextChunker` for pure text-based chunking, making it ideal for email content that doesn't require advanced document structure analysis.

## Features

- **Email Header Extraction**: Extracts Subject, From, To, Date, and Cc headers
- **Plain Text Support**: Processes plain text email bodies
- **HTML Support**: Converts HTML email bodies to plain text using BeautifulSoup
- **Multipart Email Parsing**: Handles complex multipart MIME emails
- **Text-Based Chunking**: Uses `TextChunker` for efficient sentence-based chunking
- **Configurable Processing**: Options for header inclusion and HTML extraction
- **Standard Loader Interface**: Implements `run()`, `load_as_nodes()`, and `load_as_vsfile()` methods

## Installation

The EML loader is part of the `rakam_system_vectorstore` package. Ensure you have the required dependencies:

```bash
pip install beautifulsoup4  # For HTML to text conversion
pip install chonkie==1.4.2  # For text chunking
```

## Usage

### Basic Usage

```python
from rakam_system_vectorstore.components.loader import create_eml_loader

# Create loader with default settings
loader = create_eml_loader()

# Load and chunk an EML file
chunks = loader.run("path/to/email.eml")

print(f"Extracted {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk[:100]}...")
```

### Load Without Headers

```python
# Exclude email headers from output
loader = create_eml_loader(include_headers=False)
chunks = loader.run("path/to/email.eml")
```

### Custom Chunking Parameters

```python
# Customize chunk size and overlap
loader = create_eml_loader(
    chunk_size=1024,      # Larger chunks
    chunk_overlap=100,    # More overlap
    min_sentences_per_chunk=2,
    tokenizer="character"
)
chunks = loader.run("path/to/email.eml")
```

### Load as Nodes

```python
# Load as Node objects with metadata
nodes = loader.load_as_nodes(
    "path/to/email.eml",
    custom_metadata={
        "category": "support",
        "priority": "high"
    }
)

for node in nodes:
    print(f"Node {node.metadata.position}: {len(node.content)} chars")
```

### Load as VSFile

```python
# Load as VSFile object for vector storage
vsfile = loader.load_as_vsfile("path/to/email.eml")

print(f"VSFile UUID: {vsfile.uuid}")
print(f"Number of nodes: {len(vsfile.nodes)}")
print(f"Processed: {vsfile.processed}")
```

## Configuration Options

### Chunking Parameters

- **chunk_size** (int, default: 512): Maximum tokens per chunk
- **chunk_overlap** (int, default: 50): Overlap between chunks in tokens
- **min_sentences_per_chunk** (int, default: 1): Minimum sentences per chunk
- **tokenizer** (str, default: "character"): Tokenizer to use
  - `"character"`: Character-based tokenization (fastest)
  - `"gpt2"`: GPT-2 tokenizer
  - Any HuggingFace tokenizer name

### Processing Options

- **include_headers** (bool, default: True): Include email headers in output
- **extract_html** (bool, default: True): Extract and convert HTML parts to text

## Email Structure Handling

### Headers

When `include_headers=True`, the following headers are extracted:

- Subject
- From
- To
- Date
- Cc (if present)

Headers are formatted as:

```
Subject: Email subject line
From: sender@example.com
To: recipient@example.com
Date: Thu, 02 Jan 2025 17:14:20 +0100
```

### Body Content

The loader handles different email body formats:

1. **Plain Text**: Extracted directly from `text/plain` parts
2. **HTML**: Converted to plain text from `text/html` parts (requires BeautifulSoup)
3. **Multipart**: Processes all text and HTML parts in the email

### Attachments

Attachments are currently skipped during processing. Only inline text content is extracted.

## Integration with AdaptiveLoader

The `EmlLoader` is automatically integrated with `AdaptiveLoader`, which can detect and process EML files:

```python
from rakam_system_vectorstore.components.loader import AdaptiveLoader

loader = AdaptiveLoader()
chunks = loader.run("path/to/email.eml")  # Automatically uses EmlLoader
```

## Architecture

### Text Chunking vs Advanced Chunking

The EML loader uses `TextChunker` instead of `AdvancedChunker` because:

1. **Email content is already plain text**: No need for document structure analysis
2. **Simpler processing**: Emails don't have complex layouts like PDFs
3. **Faster performance**: Text chunking is more efficient for simple text
4. **Better for conversational content**: Sentence-based chunking works well for email text

### Processing Pipeline

```
EML File → Email Parser → Header Extraction → Body Extraction → Text Chunking → Chunks
                              ↓                      ↓
                         (optional)            Plain Text + HTML
```

## Comparison with Other Loaders

| Feature | EmlLoader | PdfLoader | OdtLoader |
|---------|-----------|-----------|-----------|
| Chunker | TextChunker | AdvancedChunker | AdvancedChunker |
| Structure | Simple text | Complex layout | Document structure |
| Images | No | Yes | Yes |
| Tables | No | Yes | No |
| Speed | Fast | Slower | Medium |
| Use Case | Emails | Documents | Office docs |

## Examples

See the complete examples in:
```
rakam_systems/examples/ai_vectorstore_examples/eml_loader_example.py
```

Run the examples:
```bash
python rakam_systems/examples/ai_vectorstore_examples/eml_loader_example.py
```

## Error Handling

The loader provides clear error messages for common issues:

```python
try:
    chunks = loader.run("email.eml")
except FileNotFoundError:
    print("EML file not found")
except ValueError:
    print("File is not a valid EML")
except Exception as e:
    print(f"Processing error: {e}")
```

## Performance

- **Small emails** (<10KB): ~0.01s processing time
- **Medium emails** (10-100KB): ~0.05s processing time
- **Large emails** (>100KB): ~0.1-0.5s processing time

Processing time depends on:
- Email size
- Number of HTML parts
- Chunk size settings

## Best Practices

1. **Use smaller chunk sizes** for better granularity in email search
2. **Enable header extraction** for context about email metadata
3. **Keep HTML extraction enabled** to process rich email content
4. **Add custom metadata** to nodes for categorization and filtering
5. **Use AdaptiveLoader** for mixed file type processing

## Limitations

- Does not extract attachments
- Images in emails are not processed
- Embedded objects are skipped
- Only processes text content
- MSG format not yet supported

## Future Enhancements

Potential improvements:
- [ ] Attachment extraction and processing
- [ ] Image extraction from emails
- [ ] MSG format support
- [ ] Email thread reconstruction
- [ ] Metadata extraction (importance, flags, etc.)
- [ ] Reply chain parsing

## API Reference

### EmlLoader Class

```python
class EmlLoader(Loader):
    def __init__(self, name: str = "eml_loader", config: Optional[Dict[str, Any]] = None)
    def run(self, source: str) -> List[str]
    def load_as_nodes(self, source: Union[str, Path], source_id: Optional[str] = None, 
                      custom_metadata: Optional[Dict[str, Any]] = None) -> List[Node]
    def load_as_vsfile(self, file_path: Union[str, Path], 
                       custom_metadata: Optional[Dict[str, Any]] = None) -> VSFile
```

### Factory Function

```python
def create_eml_loader(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_sentences_per_chunk: int = 1,
    tokenizer: str = "character",
    include_headers: bool = True,
    extract_html: bool = True
) -> EmlLoader
```

## Contributing

To extend the EML loader:

1. Add new features to `eml_loader.py`
2. Update tests in the examples
3. Document changes in this README
4. Ensure compatibility with the Loader interface

## License

Part of the rakam_systems package.


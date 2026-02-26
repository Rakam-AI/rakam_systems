---
title: Vectorstore Package
---

# Vectorstore Package

The vectorstore package provides vector database solutions and document processing. Install with `pip install rakam-systems-vectorstore[all]` (requires core).

## Core Data Structures

```python
from rakam_systems_vectorstore.core import Node, NodeMetadata, VSFile

# VSFile - Represents a document source
vsfile = VSFile(file_path="/path/to/document.pdf")
print(vsfile.uuid, vsfile.file_name, vsfile.mime_type)

# NodeMetadata - Metadata for document chunks
metadata = NodeMetadata(
    source_file_uuid=str(vsfile.uuid),
    position=0,  # Page number or chunk position
    custom={"author": "John", "date": "2024-01-01"}
)

# Node - A chunk with content and metadata
node = Node(content="Document content here...", metadata=metadata)
node.embedding = [0.1, 0.2, 0.3, ...]  # Set after embedding
```

## ConfigurablePgVectorStore

Enhanced PostgreSQL vector store with full configuration support:

```python
from rakam_systems_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

# From configuration object
config = VectorStoreConfig()
store = ConfigurablePgVectorStore(config=config)

# From YAML file
store = ConfigurablePgVectorStore(config="vectorstore_config.yaml")

# From dictionary
store = ConfigurablePgVectorStore(config={
    "name": "my_store",
    "embedding": {
        "model_type": "sentence_transformer",
        "model_name": "Snowflake/snowflake-arctic-embed-m"
    },
    "search": {
        "similarity_metric": "cosine",
        "enable_hybrid_search": True,
        "hybrid_alpha": 0.7
    }
})

# Setup (initializes embedding model, database tables)
store.setup()

# Add documents
store.add_nodes(nodes)
store.add_vsfile(vsfile)

# Vector search (semantic similarity)
results = store.search("What is machine learning?", top_k=5)

# Hybrid search (combines vector + keyword search)
results = store.hybrid_search("machine learning", top_k=10, alpha=0.7)

# Keyword search (full-text search with BM25 or ts_rank)
results = store.keyword_search(
    query="machine learning algorithms",
    top_k=10,
    ranking_algorithm="bm25",  # or "ts_rank"
    k1=1.2,  # BM25 parameter
    b=0.75   # BM25 parameter
)

# Update vectors
store.update_vector(node_id, new_embedding)

# Cleanup
store.shutdown()
```

### Keyword Search

Full-text search using PostgreSQL's built-in capabilities with BM25 or ts_rank ranking:

```python
from rakam_systems_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

config = VectorStoreConfig(
    search={
        "keyword_ranking_algorithm": "bm25",  # or "ts_rank"
        "keyword_k1": 1.2,  # BM25 k1 parameter
        "keyword_b": 0.75   # BM25 b parameter
    }
)

store = ConfigurablePgVectorStore(config=config)
store.setup()

# Keyword search with BM25 ranking
results = store.keyword_search(
    query="machine learning neural networks",
    top_k=10,
    ranking_algorithm="bm25"
)

# Keyword search with ts_rank
results = store.keyword_search(
    query="deep learning",
    top_k=10,
    ranking_algorithm="ts_rank"
)

# Results include content and relevance scores
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:200]}...")
```

**Ranking Algorithms:**

- **BM25**: Best Match 25, probabilistic ranking function
  - `k1`: Term frequency saturation parameter (default: 1.2)
  - `b`: Length normalization parameter (default: 0.75)
- **ts_rank**: PostgreSQL's text search ranking function
  - Weights different parts of documents differently
  - Good for structured documents

### Multi-Model Support

Each embedding model automatically gets dedicated tables:

```python
# Using different models - each gets its own tables
store_minilm = ConfigurablePgVectorStore(config=config_minilm)
store_mpnet = ConfigurablePgVectorStore(config=config_mpnet)

# Table names are based on model names:
# - application_nodeentry_all_minilm_l6_v2
# - application_nodeentry_snowflake_arctic_embed_m

# Disable model-specific tables if needed (not recommended)
store = ConfigurablePgVectorStore(
    config=config,
    use_dimension_specific_tables=False
)
```

## ConfigurableEmbeddings

Multi-backend embedding model with unified interface:

```python
from rakam_systems_vectorstore import ConfigurableEmbeddings, create_embedding_model

# Using Sentence Transformers (local)
embeddings = ConfigurableEmbeddings(config={
    "model_type": "sentence_transformer",
    "model_name": "Snowflake/snowflake-arctic-embed-m",
    "batch_size": 128,
    "normalize": True
})

# Using OpenAI (with batch processing)
embeddings = ConfigurableEmbeddings(config={
    "model_type": "openai",
    "model_name": "text-embedding-3-small",
    "api_key": "...",  # Or use OPENAI_API_KEY
    "batch_size": 100   # OpenAI supports larger batches
})

# Using Cohere
embeddings = ConfigurableEmbeddings(config={
    "model_type": "cohere",
    "model_name": "embed-english-v3.0",
    "api_key": "..."  # Or use COHERE_API_KEY
})

# Using HuggingFace models with authentication
embeddings = ConfigurableEmbeddings(config={
    "model_type": "sentence_transformer",
    "model_name": "private/model-name",
    # Uses HUGGINGFACE_TOKEN environment variable
})

embeddings.setup()

# Encode texts with automatic batch processing
vectors = embeddings.run(["Hello world", "How are you?"])

# Encode large datasets with progress tracking
large_texts = ["text" + str(i) for i in range(10000)]
vectors = embeddings.run(large_texts)  # Shows progress bar

# Encode queries (optimized for single texts)
query_vector = embeddings.encode_query("What is AI?")

# Encode documents (optimized for batches)
doc_vectors = embeddings.encode_documents(documents)

# Get dimension
dim = embeddings.embedding_dimension
```

**Performance Features:**

- Automatic batch processing with progress tracking
- Memory optimization with garbage collection
- Token truncation for oversized texts
- Mini-batch processing for large datasets
- CUDA memory management for GPU acceleration

### Factory Function

```python
embeddings = create_embedding_model(
    model_type="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
    batch_size=64
)
```

## AdaptiveLoader

Automatically detects and processes various file types:

```python
from rakam_systems_vectorstore import AdaptiveLoader, create_adaptive_loader

loader = AdaptiveLoader(config={
    "encoding": "utf-8",
    "chunk_size": 512,
    "chunk_overlap": 50
})

# Supported file types:
# - Text: .txt, .text
# - Markdown: .md, .markdown
# - Documents: .pdf, .docx, .doc, .odt
# - Email: .eml, .msg
# - Data: .json, .csv, .tsv, .xlsx, .xls
# - HTML: .html, .htm, .xhtml
# - Code: .py, .js, .ts, .java, .cpp, .go, .rs, .rb, etc.

# Load as single text
text = loader.load_as_text("document.pdf")

# Load as chunks
chunks = loader.load_as_chunks("document.pdf")

# Load as nodes (with metadata)
nodes = loader.load_as_nodes("document.pdf", custom_metadata={"category": "science"})

# Load as VSFile
vsfile = loader.load_as_vsfile("document.pdf")

# Also handles raw text
chunks = loader.load_as_chunks("This is raw text content...")
```

### Factory Function

```python
loader = create_adaptive_loader(
    chunk_size=1024,
    chunk_overlap=100,
    encoding='utf-8'
)
```

## Specialized Loaders

Located in `rakam_systems_vectorstore/components/loader/`:

| Loader           | File Types              | Features                                                                           |
| ---------------- | ----------------------- | ---------------------------------------------------------------------------------- |
| `PdfLoader`      | `.pdf`                  | Advanced PDF processing with Docling, image extraction, table detection            |
| `PdfLoaderLight` | `.pdf`                  | Lightweight PDF processing with pymupdf4llm, markdown conversion, image extraction |
| `DocLoader`      | `.docx`, `.doc`         | Microsoft Word documents, image extraction                                         |
| `OdtLoader`      | `.odt`                  | OpenDocument Text, image extraction                                                |
| `MdLoader`       | `.md`                   | Markdown with structure preservation, YAML frontmatter                             |
| `HtmlLoader`     | `.html`, `.htm`         | HTML parsing and text extraction                                                   |
| `EmlLoader`      | `.eml`, `.msg`          | Email files (loaded as single nodes)                                               |
| `TabularLoader`  | `.csv`, `.tsv`, `.xlsx` | Tabular data processing, preserves column structure                                |
| `CodeLoader`     | `.py`, `.js`, etc.      | Code-aware chunking with syntax preservation                                       |

### PdfLoaderLight

A lightweight alternative to PdfLoader using pymupdf4llm for efficient PDF processing:

```python
from rakam_systems_vectorstore.components.loader import PdfLoaderLight

loader = PdfLoaderLight(
    name="pdf_loader_light",
    config={
        "chunk_size": 512,
        "chunk_overlap": 50,
        "extract_images": True,
        "image_path": "./extracted_images",
        "page_chunks": True,  # Create one chunk per page
        "write_images": True  # Save images to disk
    }
)

# Load as markdown
markdown_text = loader.load_as_text("document.pdf")

# Load as chunks (one per page or custom chunking)
chunks = loader.load_as_chunks("document.pdf")

# Load as nodes with metadata
nodes = loader.load_as_nodes("document.pdf")

# Access extracted images
image_paths = loader.get_image_paths()
for img_id, img_path in image_paths.items():
    print(f"Image {img_id}: {img_path}")
```

**Key Features:**

- Fast PDF to markdown conversion
- Optional image extraction and saving
- Page-aware chunking
- Thread-safe operations
- Lower memory footprint than PdfLoader

### Image Extraction Support

Multiple loaders now support image extraction:

```python
from rakam_systems_vectorstore.components.loader import DocLoader, OdtLoader, PdfLoaderLight

# DocLoader with image extraction
doc_loader = DocLoader(config={
    "extract_images": True,
    "image_path": "./doc_images"
})
nodes = doc_loader.load_as_nodes("document.docx")

# Access extracted images
for img_id, img_path in doc_loader.get_image_paths().items():
    print(f"Image {img_id}: {img_path}")

# OdtLoader with image extraction
odt_loader = OdtLoader(config={
    "extract_images": True,
    "image_path": "./odt_images"
})
nodes = odt_loader.load_as_nodes("document.odt")

# PdfLoaderLight with image extraction
pdf_loader = PdfLoaderLight(config={
    "extract_images": True,
    "image_path": "./pdf_images",
    "write_images": True
})
nodes = pdf_loader.load_as_nodes("document.pdf")
```

## TextChunker

Sentence-based text chunking using Chonkie:

```python
from rakam_systems_vectorstore.components.chunker import TextChunker, create_text_chunker

chunker = TextChunker(
    chunk_size=512,        # Tokens per chunk
    chunk_overlap=50,      # Overlap in tokens
    min_sentences_per_chunk=1,
    tokenizer="character"  # Or "gpt2", HuggingFace tokenizer
)

chunks = chunker.chunk_text("Long document text...")
# Returns: [{"text": "...", "token_count": 100, "start_index": 0, "end_index": 500}, ...]

# Process multiple documents
all_chunks = chunker.run(["doc1 text", "doc2 text"])
```

## AdvancedChunker

Advanced document chunking using Docling for context-aware chunking with heading preservation:

```python
from rakam_systems_vectorstore.components.chunker import AdvancedChunker

chunker = AdvancedChunker(
    name="advanced_chunker",
    config={
        "max_tokens": 512,           # Maximum tokens per chunk
        "merge_peers": True,          # Merge peer sections
        "min_chunk_tokens": 64,       # Minimum tokens per chunk
        "filter_toc": True,           # Filter table of contents
        "include_heading_markers": True  # Include markdown headings
    }
)

# Chunk text with context preservation
chunks = chunker.chunk_text("Document text with headings...")

# Each chunk includes:
# - text: The chunk content
# - token_count: Number of tokens
# - start_index: Starting position
# - end_index: Ending position
# - heading_context: Hierarchical heading information

# Process with heading markers
chunker_with_markers = AdvancedChunker(config={
    "include_heading_markers": True
})
chunks = chunker_with_markers.chunk_text("""
# Main Title
## Section 1
Content here...
## Section 2
More content...
""")
# Output includes markdown-style headings in chunks
```

**Key Features:**

- Context-aware chunking with heading hierarchy
- Automatic merging of small chunks
- Table of contents filtering
- Image and table fragment handling
- Markdown heading markers support
- Configurable token limits and merging behavior

---
title: Build vector pipelines
---

# Build vector pipelines

The vectorstore package provides document processing, embedding, and vector storage. Install with `pip install rakam-systems-vectorstore[all]` (requires core).

## Core data structures

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

## Embeddings

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
    "batch_size": 100
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

# Encode queries (optimized for single texts)
query_vector = embeddings.encode_query("What is AI?")

# Encode documents (optimized for batches)
doc_vectors = embeddings.encode_documents(documents)

# Get dimension
dim = embeddings.embedding_dimension
```

Performance features: automatic batch processing with progress tracking, memory optimization with garbage collection, token truncation for oversized texts, CUDA memory management for GPU acceleration.

### Factory function

```python
embeddings = create_embedding_model(
    model_type="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
    batch_size=64
)
```

### AI gateway embedder (`build_embedder`)

`ConfigurableEmbeddings` above is configured by backend. The AI gateway takes the
other route: one `"<provider>:<model>"` reference resolved through pydantic-ai, the
same grammar chat models use. Use it when the provider is a deployment choice
rather than a code choice — see [the AI gateway spec](../specs/ai-gateway.md).

```python
from rakam_systems_core.config_schema import EmbeddingRef
from rakam_systems_vectorstore import build_embedder

embedder = build_embedder(EmbeddingRef(ref="openai:text-embedding-3-small", dim=1536))
vectors = embedder.run(["Hello world", "How are you?"])   # sync, for the stores
vectors = await embedder.arun(["Hello world"])            # inside an event loop
```

`EmbeddingRef` carries the ref, a required `dim`, and an optional `base_url`.
`dim` is not documentation: it is sent to the provider as the requested output
width *and* checked against what comes back, so a model that silently returns a
different width fails loudly instead of corrupting the index.

`build_embedder(cfg, batch_size=N)` splits the corpus into `N`-sized provider
calls, in input order. The default (`None`) sends everything in one call.

Three routing cases:

| Ref | Goes to |
|---|---|
| `sentence-transformers:<model>` (or `st:`, `local:`) | `ConfigurableEmbeddings`, offline — pydantic-ai has no local ST embedder |
| `mistral:mistral-embed` | This package's `MistralEmbeddingModel` — pydantic-ai ships no Mistral embeddings backend, so `infer_embedding_model` would raise `UserError` |
| anything else | pydantic-ai's `infer_embedding_model` (OpenAI, Azure, Cohere, Google, Bedrock, Voyage, …) |

The first two are a closed exception list, not the start of a provider registry:
a provider only belongs there while pydantic-ai ships no backend for it.

```python
# Mistral. Needs MISTRAL_API_KEY and the `mistral` extra
# (pip install "rakam-systems-vectorstore[mistral]").
embedder = build_embedder(EmbeddingRef(ref="mistral:mistral-embed", dim=1024))

# A self-hosted or proxied endpoint. Give the ORIGIN: mistralai appends /v1/...
# itself, and a trailing /v1 is stripped for you, so this matches how base_url is
# written for the OpenAI-compatible route.
embedder = build_embedder(
    EmbeddingRef(ref="mistral:mistral-embed", base_url="https://gw.internal", dim=1024)
)
```

Note the ref must go through `build_embedder`. Handing `"mistral:mistral-embed"`
straight to pydantic-ai's `infer_embedding_model` still raises
`UserError: Unknown embeddings model`.

## Document loading

### AdaptiveLoader

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

Factory function:

```python
loader = create_adaptive_loader(
    chunk_size=1024,
    chunk_overlap=100,
    encoding='utf-8'
)
```

### Specialized loaders

Located in `rakam_systems_vectorstore/components/loader/`:

| Loader           | File types              | Features                                                                           |
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

A lightweight alternative to PdfLoader using pymupdf4llm:

```python
from rakam_systems_vectorstore.components.loader import PdfLoaderLight

loader = PdfLoaderLight(
    name="pdf_loader_light",
    config={
        "chunk_size": 512,
        "chunk_overlap": 50,
        "extract_images": True,
        "image_path": "./extracted_images",
        "page_chunks": True,
        "write_images": True
    }
)

markdown_text = loader.load_as_text("document.pdf")
chunks = loader.load_as_chunks("document.pdf")
nodes = loader.load_as_nodes("document.pdf")

# Access extracted images
image_paths = loader.get_image_paths()
for img_id, img_path in image_paths.items():
    print(f"Image {img_id}: {img_path}")
```

### Image extraction support

Multiple loaders support image extraction:

```python
from rakam_systems_vectorstore.components.loader import DocLoader, OdtLoader, PdfLoaderLight

doc_loader = DocLoader(config={
    "extract_images": True,
    "image_path": "./doc_images"
})
nodes = doc_loader.load_as_nodes("document.docx")

for img_id, img_path in doc_loader.get_image_paths().items():
    print(f"Image {img_id}: {img_path}")
```

## Chunking

### TextChunker

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

### AdvancedChunker

Context-aware chunking using Docling with heading preservation:

```python
from rakam_systems_vectorstore.components.chunker import AdvancedChunker

chunker = AdvancedChunker(
    name="advanced_chunker",
    config={
        "max_tokens": 512,
        "merge_peers": True,
        "min_chunk_tokens": 64,
        "filter_toc": True,
        "include_heading_markers": True
    }
)

chunks = chunker.chunk_text("Document text with headings...")
# Each chunk includes: text, token_count, start_index, end_index, heading_context
```

Features: context-aware chunking with heading hierarchy, automatic merging of small chunks, table of contents filtering, image and table fragment handling, markdown heading markers support.

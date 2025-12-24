# PDF Loader Architecture

## Class Hierarchy

```
BaseComponent (framework base)
    │
    └── Loader (abstract interface)
            │
            ├── FileLoader (simple stub)
            │
            ├── AdaptiveLoader (multi-format loader)
            │
            └── PdfLoader ✨ (new - specialized PDF processor)
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Input: PDF File                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    PdfLoader.run()                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. Validate file is PDF (extension + MIME type)        │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 2. Convert with Docling (DocumentConverter)            │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 3. Extract & Save Images (to data/scratch/)            │ │
│  │    • Page images (.png)                                │ │
│  │    • Table images (.png)                               │ │
│  │    • Picture/figure images (.png)                      │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 4. Save Markdown/HTML (to data/scratch/)               │ │
│  │    • With embedded images (.md)                        │ │
│  │    • With referenced images (.md, .html)               │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 5. Extract Text & Chunk (TextChunker)                  │ │
│  │    • Per-page markdown export                          │ │
│  │    • Character-based chunking with overlap             │ │
│  │    • Word-boundary aware splitting                     │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
    ┌───────────────────┐   ┌──────────────────────┐
    │  List[str]        │   │  Alternative Methods │
    │  (text chunks)    │   │  ─────────────────── │
    └───────────────────┘   │  load_as_nodes()     │
                            │  → List[Node]        │
                            │                      │
                            │  load_as_vsfile()    │
                            │  → VSFile            │
                            └──────────────────────┘
```

## Component Interactions

```
┌──────────────────────────────────────────────────────────────┐
│                     Application Layer                         │
│  (Your code that needs to load PDFs)                         │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│                        PdfLoader                              │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ run()          │  │load_as_nodes()│  │load_as_vsfile() │ │
│  │ → List[str]    │  │ → List[Node] │  │ → VSFile        │ │
│  └────────────────┘  └──────────────┘  └─────────────────┘ │
└──┬─────────────────────────┬───────────────────┬────────────┘
   │                         │                   │
   │ uses                    │ uses              │ creates
   │                         │                   │
   ▼                         ▼                   ▼
┌────────────────┐  ┌──────────────────┐  ┌──────────────┐
│  TextChunker   │  │  Node            │  │  VSFile      │
│  (chunking)    │  │  NodeMetadata    │  │  (wrapper)   │
└────────────────┘  └──────────────────┘  └──────────────┘
   │
   │ uses
   ▼
┌────────────────────────────────────────────────────────────┐
│                     Docling Library                         │
│  ┌──────────────────┐  ┌──────────────────────────────┐   │
│  │DocumentConverter │  │  PdfPipelineOptions          │   │
│  │                  │  │  • images_scale              │   │
│  │ • convert()      │  │  • generate_page_images      │   │
│  │ • extracts text  │  │  • generate_picture_images   │   │
│  │ • extracts images│  └──────────────────────────────┘   │
│  └──────────────────┘                                      │
└────────────────────────────────────────────────────────────┘
```

## File System Structure

```
project/
├── app/
│   └── rakam_systems/
│       └── rakam_systems/
│           └── ai_vectorstore/
│               └── components/
│                   └── loader/
│                       ├── __init__.py (exports PdfLoader)
│                       ├── file_loader.py
│                       ├── adaptive_loader.py
│                       └── pdf_loader.py ✨ (new)
│
├── data/
│   ├── pdf_example/
│   │   └── document.pdf (input)
│   │
│   └── scratch/ ✨ (created by loader)
│       ├── document-page-1.png
│       ├── document-page-2.png
│       ├── document-table-1.png
│       ├── document-picture-1.png
│       ├── document-with-images.md
│       ├── document-with-image-refs.md
│       └── document-with-image-refs.html
│
└── test_pdf_loader.py (test script)
```

## Processing Pipeline Detail

### Phase 1: Validation
```
Input PDF
    │
    ├─→ Check file exists ──────────→ FileNotFoundError if missing
    │
    ├─→ Check .pdf extension ───────→ ValueError if wrong extension
    │
    └─→ Check MIME type ────────────→ ValueError if not application/pdf
```

### Phase 2: Conversion (Docling)
```
Valid PDF
    │
    └─→ DocumentConverter.convert()
            │
            ├─→ Parse document structure
            ├─→ Extract text with layout
            ├─→ Render page images (if enabled)
            ├─→ Extract embedded pictures
            └─→ Extract tables
            │
            └─→ ConversionResult
```

### Phase 3: Image Extraction
```
ConversionResult
    │
    ├─→ Page Images
    │   └─→ Save as: {filename}-page-{N}.png
    │
    ├─→ Table Images
    │   └─→ Save as: {filename}-table-{N}.png
    │
    └─→ Picture Images
        └─→ Save as: {filename}-picture-{N}.png
```

### Phase 4: Markdown Export
```
ConversionResult
    │
    ├─→ Markdown (embedded images)
    │   └─→ Save as: {filename}-with-images.md
    │
    ├─→ Markdown (referenced images)
    │   └─→ Save as: {filename}-with-image-refs.md
    │
    └─→ HTML (referenced images)
        └─→ Save as: {filename}-with-image-refs.html
```

### Phase 5: Text Chunking
```
ConversionResult
    │
    └─→ For each page:
            │
            ├─→ Export page as markdown
            │
            └─→ TextChunker.chunk_text()
                    │
                    ├─→ Split into chunks (512 chars)
                    ├─→ Apply overlap (50 chars)
                    ├─→ Break at word boundaries
                    │
                    └─→ Text Chunks
```

## Configuration Options

```python
PdfLoader Configuration
    │
    ├── image_scale: float = 2.0
    │   └─→ Controls rendered image resolution
    │       (1.0 = 72 DPI, 2.0 = 144 DPI, 3.0 = 216 DPI)
    │
    ├── generate_page_images: bool = True
    │   └─→ Whether to generate full page images
    │
    ├── generate_picture_images: bool = True
    │   └─→ Whether to extract embedded pictures/figures
    │
    ├── chunk_size: int = 512
    │   └─→ Size of text chunks in characters
    │
    ├── chunk_overlap: int = 50
    │   └─→ Overlap between adjacent chunks
    │
    ├── save_images: bool = True
    │   └─→ Whether to save extracted images to disk
    │
    ├── save_markdown: bool = True
    │   └─→ Whether to save markdown/HTML files
    │
    └── scratch_folder_name: str = 'scratch'
        └─→ Name of output folder (in data directory)
```

## Integration Points

### 1. Vector Store Integration
```python
# Load PDF as VSFile
loader = PdfLoader()
vsfile = loader.load_as_vsfile("document.pdf")

# Add to vector store
vector_store.add_nodes(vsfile.nodes)
```

### 2. Direct Text Access
```python
# Get text chunks directly
loader = PdfLoader()
chunks = loader.run("document.pdf")

# Process chunks
for chunk in chunks:
    process(chunk)
```

### 3. Custom Metadata
```python
# Attach metadata to nodes
loader = PdfLoader()
nodes = loader.load_as_nodes(
    "document.pdf",
    custom_metadata={
        "category": "technical",
        "priority": "high",
        "department": "engineering"
    }
)
```

## Error Handling Flow

```
PdfLoader.run(source)
    │
    ├─→ File not found? ──────────→ raise FileNotFoundError
    │
    ├─→ Not a PDF? ───────────────→ raise ValueError
    │
    ├─→ Conversion fails? ────────→ raise Exception (with context)
    │
    ├─→ Image save fails? ────────→ log warning, continue
    │
    ├─→ Markdown save fails? ─────→ log warning, continue
    │
    └─→ Success ──────────────────→ return chunks
```

## Performance Characteristics

```
Input: PDF Document
    │
    ├─→ Small (1-5 pages)
    │   └─→ ~1-3 seconds
    │
    ├─→ Medium (10-50 pages)
    │   └─→ ~5-15 seconds
    │
    └─→ Large (100+ pages)
        └─→ ~30-60+ seconds

Factors affecting performance:
    • Number of pages
    • Number of images
    • Image resolution (image_scale)
    • Complex layouts
    • File size
```

## Thread Safety

```
PdfLoader instance
    │
    ├─→ DocumentConverter: ⚠️  Not guaranteed thread-safe
    │
    ├─→ TextChunker: ✅ Thread-safe (stateless)
    │
    └─→ Recommendation: Create separate PdfLoader instance per thread
```

## Memory Considerations

```
Memory Usage Components:
    │
    ├─→ Loaded PDF (in Docling): ~10-50MB per document
    │
    ├─→ Rendered images: ~1-5MB per page (depends on image_scale)
    │
    ├─→ Text chunks: ~100KB-1MB (depends on document size)
    │
    └─→ Total: ~10-100MB per document (varies widely)

Memory optimization:
    • Process documents one at a time
    • Clear references after processing
    • Consider lowering image_scale for large batches
```


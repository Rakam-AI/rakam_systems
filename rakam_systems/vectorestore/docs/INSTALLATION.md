# AI Vectorstore - Installation Guide

This guide explains how to install the `ai_vectorstore` module either as a standalone package or as part of the `rakam-systems` framework.

## Table of Contents

- [Installation Options](#installation-options)
- [Standalone Installation](#standalone-installation)
- [As Part of Rakam Systems](#as-part-of-rakam-systems)
- [Optional Dependencies](#optional-dependencies)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Installation Options

The `ai_vectorstore` module can be installed in several ways depending on your needs:

### 1. Standalone Installation (Recommended for Vector Store Only)

If you only need the vector store functionality without the full Rakam Systems framework:

```bash
# From the ai_vectorstore directory
cd app/rakam_systems/rakam_systems/ai_vectorstore
pip install -e .
```

Or install directly from the main rakam_systems directory:

```bash
cd app/rakam_systems
pip install -e "./rakam_systems/ai_vectorstore"
```

### 2. As Part of Rakam Systems

If you want the full Rakam Systems framework with ai_vectorstore as an optional component:

```bash
cd app/rakam_systems
pip install -e ".[ai-vectorstore]"
```

## Standalone Installation

### Basic Installation (Core Only)

Install just the core vectorstore functionality:

```bash
cd app/rakam_systems/rakam_systems/ai_vectorstore
pip install -e .
```

This installs:
- Core data structures (Node, VSFile, NodeMetadata)
- Configuration system
- Base interfaces

### With PostgreSQL Backend

For production use with PostgreSQL and pgvector:

```bash
pip install -e ".[postgres]"
```

This adds:
- `psycopg2-binary` - PostgreSQL adapter
- `django` - ORM for database management

### With FAISS Backend

For in-memory, high-performance vector search:

```bash
pip install -e ".[faiss]"
```

This adds:
- `faiss-cpu` - Facebook AI Similarity Search

### With Local Embeddings

To use local embedding models (SentenceTransformers):

```bash
pip install -e ".[local-embeddings]"
```

This adds:
- `sentence-transformers` - Local embedding models
- `torch` - PyTorch for model inference

### With OpenAI Embeddings

To use OpenAI's embedding API:

```bash
pip install -e ".[openai]"
```

This adds:
- `openai` - OpenAI API client

### With Cohere Embeddings

To use Cohere's embedding API:

```bash
pip install -e ".[cohere]"
```

This adds:
- `cohere` - Cohere API client

### With Document Loaders

To support various document formats (PDF, DOCX, HTML, etc.):

```bash
pip install -e ".[loaders]"
```

This adds:
- `pymupdf` - PDF processing
- `python-docx` - Word document processing
- `beautifulsoup4` - HTML parsing
- `python-magic` - File type detection

### Complete Installation

Install everything (all backends and features):

```bash
pip install -e ".[all]"
```

### Development Installation

For development with testing and linting tools:

```bash
pip install -e ".[all,dev]"
```

## As Part of Rakam Systems

### Installing as an Optional Dependency

From the main `rakam_systems` directory:

```bash
cd app/rakam_systems

# Install rakam-systems with ai-vectorstore
pip install -e ".[ai-vectorstore]"
```

This installs the complete rakam-systems framework plus all ai-vectorstore dependencies.

### Multiple Optional Dependencies

You can combine multiple optional dependencies:

```bash
# Install with multiple components
pip install -e ".[ai-vectorstore,ai-agents,llm-gateway]"

# Install everything
pip install -e ".[all]"
```

## Optional Dependencies

### Combining Extras

You can combine multiple extras in a single installation:

```bash
# PostgreSQL + Local embeddings + Document loaders
pip install -e ".[postgres,local-embeddings,loaders]"

# FAISS + OpenAI embeddings
pip install -e ".[faiss,openai]"

# Everything
pip install -e ".[all]"
```

### Available Extras

| Extra | Description | Key Dependencies |
|-------|-------------|------------------|
| `postgres` | PostgreSQL backend with Django ORM | psycopg2-binary, django |
| `faiss` | FAISS in-memory vector store | faiss-cpu |
| `local-embeddings` | Local embedding models | sentence-transformers, torch |
| `openai` | OpenAI embeddings API | openai |
| `cohere` | Cohere embeddings API | cohere |
| `loaders` | Document loaders for various formats | pymupdf, python-docx, beautifulsoup4 |
| `all` | Everything above | All of the above |
| `dev` | Development tools | pytest, black, ruff |

## Verification

After installation, verify that the package is correctly installed:

```python
# Test core imports
from rakam_systems.ai_vectorstore import Node, NodeMetadata, VSFile
from rakam_systems.ai_vectorstore.config import VectorStoreConfig

print("Core imports successful!")

# Test configuration system
config = VectorStoreConfig()
print(f"Default config loaded: {config.name}")

# Test optional imports (if installed)
try:
    from rakam_systems.ai_vectorstore import ConfigurablePgVectorStore
    print("PostgreSQL backend available!")
except ImportError:
    print("PostgreSQL backend not installed")

try:
    from rakam_systems.ai_vectorstore import FaissVectorStore
    print("FAISS backend available!")
except ImportError:
    print("FAISS backend not installed")

try:
    from rakam_systems.ai_vectorstore.components.embedding_model.configurable_embeddings import ConfigurableEmbeddings
    print("Embedding models available!")
except ImportError:
    print("Embedding models not installed")
```

### Quick Test Script

Create a file `test_installation.py`:

```python
#!/usr/bin/env python3
"""Quick test to verify ai_vectorstore installation."""

def test_core():
    """Test core functionality."""
    from rakam_systems.ai_vectorstore import Node, NodeMetadata, VSFile
    from rakam_systems.ai_vectorstore.config import VectorStoreConfig
    
    # Create a sample node
    node = Node(
        content="Test content",
        metadata=NodeMetadata(
            source_file_uuid="test-123",
            position=0
        )
    )
    
    # Load default config
    config = VectorStoreConfig()
    
    print("✓ Core functionality working")
    return True

def test_postgres():
    """Test PostgreSQL backend."""
    try:
        from rakam_systems.ai_vectorstore import ConfigurablePgVectorStore
        print("✓ PostgreSQL backend available")
        return True
    except ImportError as e:
        print(f"✗ PostgreSQL backend not available: {e}")
        return False

def test_faiss():
    """Test FAISS backend."""
    try:
        from rakam_systems.ai_vectorstore import FaissVectorStore
        print("✓ FAISS backend available")
        return True
    except ImportError as e:
        print(f"✗ FAISS backend not available: {e}")
        return False

def test_embeddings():
    """Test embedding models."""
    try:
        from rakam_systems.ai_vectorstore.components.embedding_model.configurable_embeddings import ConfigurableEmbeddings
        print("✓ Embedding models available")
        return True
    except ImportError as e:
        print(f"✗ Embedding models not available: {e}")
        return False

if __name__ == "__main__":
    print("Testing ai_vectorstore installation...\n")
    
    test_core()
    test_postgres()
    test_faiss()
    test_embeddings()
    
    print("\nInstallation test complete!")
```

Run it:

```bash
python test_installation.py
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'rakam_systems'`

**Solution**: Make sure you installed the package in editable mode from the correct directory:

```bash
# For standalone installation
cd app/rakam_systems/rakam_systems/ai_vectorstore
pip install -e .

# For rakam-systems installation
cd app/rakam_systems
pip install -e ".[ai-vectorstore]"
```

#### 2. Django Not Found

**Problem**: `ModuleNotFoundError: No module named 'django'`

**Solution**: Install with PostgreSQL support:

```bash
pip install -e ".[postgres]"
```

#### 3. Sentence Transformers Import Error

**Problem**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution**: Install with local embeddings support:

```bash
pip install -e ".[local-embeddings]"
```

#### 4. FAISS Not Available

**Problem**: `ModuleNotFoundError: No module named 'faiss'`

**Solution**: Install FAISS support:

```bash
pip install -e ".[faiss]"
```

For GPU support (if you have CUDA):

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

#### 5. Torch/PyTorch Issues

**Problem**: PyTorch installation fails or is very slow

**Solution**: Install PyTorch separately first with the appropriate version for your system:

```bash
# CPU only (faster download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install ai_vectorstore
pip install -e ".[local-embeddings]"
```

#### 6. PostgreSQL Connection Issues

**Problem**: Cannot connect to PostgreSQL

**Solution**: 
1. Make sure PostgreSQL is running with pgvector extension
2. Set environment variables:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=vectorstore_db
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
```

3. Or use configuration file (see `config.py` documentation)

#### 7. Package Not Found in Editable Mode

**Problem**: Changes to code not reflected after `pip install -e .`

**Solution**: 
1. Uninstall and reinstall:

```bash
pip uninstall rakam-systems-ai-vectorstore
pip install -e .
```

2. Check that you're in the correct directory
3. Verify `PYTHONPATH` if using custom paths

### Getting Help

If you encounter issues:

1. **Check the documentation**: Review the main README.md for usage examples
2. **Check dependencies**: Run `pip list | grep -E "rakam|django|faiss|sentence|torch"`
3. **Check Python version**: Requires Python 3.10+
4. **Open an issue**: Visit [GitHub Issues](https://github.com/Rakam-AI/rakam_systems/issues)

### System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: 
  - Minimum: 4GB (core only)
  - Recommended: 8GB+ (with local embeddings)
  - PostgreSQL: 8GB+ (for production)
- **Disk Space**:
  - Core: ~100MB
  - With local embeddings: ~2-5GB (model downloads)
  - PostgreSQL: Varies based on data size

## Development Setup

For contributing to ai_vectorstore:

```bash
# Clone the repository
git clone https://github.com/Rakam-AI/rakam_systems.git
cd rakam_systems/app/rakam_systems/rakam_systems/ai_vectorstore

# Install in development mode with all dependencies
pip install -e ".[all,dev]"

# Run tests (if available)
pytest

# Format code
black .

# Lint code
ruff check .
```

## Next Steps

After installation, check out:

- [README.md](README.md) - Comprehensive usage guide
- [Examples](../../examples/ai_vectorstore_examples/) - Working code examples
- [Configuration Guide](README.md#configuration) - Configuration system documentation

## Quick Start After Installation

```python
from rakam_systems.ai_vectorstore import ConfigurablePgVectorStore, Node, NodeMetadata

# Initialize vector store
store = ConfigurablePgVectorStore(name="my_store")
store.setup()

# Create collection
store.get_or_create_collection("documents")

# Add documents
nodes = [
    Node(
        content="Your document content",
        metadata=NodeMetadata(
            source_file_uuid="doc-1",
            position=0
        )
    )
]
store.add_nodes("documents", nodes)

# Search
results, nodes = store.search(
    collection_name="documents",
    query="your query"
)

print(f"Found {len(nodes)} results")
```

---

**Note**: This package is part of the Rakam Systems framework. For the complete framework, install `rakam-systems[all]`.


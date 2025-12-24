# Rakam Systems - Installation Guide

Welcome to the Rakam Systems installation guide. This document explains how to install the framework with its modular components using modern Python packaging standards.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Modular Installation](#modular-installation)
- [Installation Recipes](#installation-recipes)
- [Dependency Management](#dependency-management)
- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Migration Guide](#migration-guide)

## 🚀 Quick Start

### Install All Features

```bash
cd app/rakam_systems
pip install -e ".[all]"
```

This installs all feature modules except development tools.

### Install Everything (Including Dev Tools)

```bash
pip install -e ".[complete]"
```

This installs all features plus development tools (pytest, black, ruff, etc.).

### Install Core Package Only

```bash
pip install -e .
```

This installs only the minimal core dependencies:

- pydantic (data validation)
- pyyaml (configuration)
- python-dotenv (environment variables)
- colorlog (colored logging)
- requests (HTTP client)

## 📦 Modular Installation

Rakam Systems uses a modular architecture. Install only what you need:

### 1. AI Vectorstore

```bash
pip install -e ".[ai-vectorstore]"
```

**Includes:**

- sentence-transformers (local embeddings)
- faiss-cpu (vector search)
- psycopg2-binary (PostgreSQL)
- django (ORM for pgvector)
- Document loaders (PDF, DOCX, ODT, HTML)
- docling (advanced document processing)
- chonkie (text chunking)
- OpenAI & Cohere embeddings support

**Use cases:**

- Semantic search applications
- RAG (Retrieval-Augmented Generation)
- Document Q&A systems
- Knowledge base management

### 2. AI Agents

```bash
pip install -e ".[ai-agents]"
```

**Includes:**

- pydantic-ai (agent framework)
- mistralai (Mistral provider)
- tiktoken (tokenization)

**Use cases:**

- Building AI agents
- Tool-calling LLM applications
- Multi-step reasoning systems
- Conversational AI

### 3. LLM Gateway

```bash
pip install -e ".[llm-gateway]"
```

**Includes:**

- openai (OpenAI provider)
- mistralai (Mistral provider)
- tiktoken (tokenization)

**Use cases:**

- Multi-provider LLM applications
- Chat interfaces
- Text generation
- LLM abstraction layer

### 4. Document Loaders

```bash
pip install -e ".[loaders]"
```

**Includes:**

- beautifulsoup4 (HTML parsing)
- python-docx (DOCX files)
- pymupdf (PDF files)
- python-magic (file type detection)
- playwright (web scraping)
- odfpy (ODT files)

**Use cases:**

- Document processing pipelines
- Data ingestion
- Web scraping

### 5. Extra Features

```bash
pip install -e ".[extra]"
```

**Includes:**

- channels (Django channels)
- drf-spectacular (API documentation)

### 6. Development Tools

```bash
pip install -e ".[dev]"
```

**Includes:**

- pre-commit (git hooks)
- pytest (testing)
- pytest-asyncio (async testing)
- black (code formatting)
- ruff (linting)

### Combining Multiple Groups

```bash
pip install -e ".[ai-agents,ai-vectorstore,dev]"
```

## 🎯 Installation Recipes

### Recipe 1: RAG Application with PostgreSQL

For a production RAG application using PostgreSQL:

```bash
cd app/rakam_systems
pip install -e ".[ai-vectorstore]"
```

This installs everything needed for a RAG pipeline with persistent storage.

### Recipe 2: Agent with Vector Memory

For an AI agent with vector store memory:

```bash
pip install -e ".[ai-agents,ai-vectorstore]"
```

### Recipe 3: LLM Gateway Only

For a lightweight LLM abstraction layer:

```bash
pip install -e ".[llm-gateway]"
```

### Recipe 4: Complete Development Setup

Everything for development:

```bash
pip install -e ".[complete]"
```

## 📚 Dependency Management

### Primary Configuration: pyproject.toml ⭐

The `pyproject.toml` file is the **single source of truth** for all dependencies. It follows modern Python packaging standards (PEP 621) and provides:

- **Modular dependencies**: Install only what you need
- **Version ranges**: Flexible but safe version constraints
- **Clear organization**: Dependencies grouped by feature

### Locked Versions: requirements.txt

The `requirements.txt` file provides locked versions for reproducible installations:

```bash
pip install -r requirements.txt
```

**⚠️ Important:** This file is auto-generated from `pyproject.toml` and should NOT be edited manually.

#### Regenerating requirements.txt

When dependencies change in `pyproject.toml`:

```bash
# Install all features
pip install -e ".[complete]"

# Generate locked requirements
pip freeze > requirements.txt

# Add header comment
echo "# Auto-generated from pyproject.toml - DO NOT EDIT MANUALLY" | cat - requirements.txt > temp && mv temp requirements.txt
```

### Deprecated: setup.cfg ⚠️

The `setup.cfg` file has been **deprecated** and is kept only for backwards compatibility. All dependency management should use `pyproject.toml`.

**Migration:** If you were using `setup.cfg`, please switch to `pyproject.toml` (see [Migration Guide](#migration-guide) below).

## 💻 System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **RAM**: 4GB minimum (8GB+ recommended)
- **Disk**: 1GB free space (more for models)
- **OS**: Linux, macOS, or Windows

### For AI Vectorstore with Local Embeddings

- **RAM**: 8GB+ recommended
- **Disk**: 5GB+ (for model downloads)
- **GPU**: Optional, but recommended for faster inference

### For PostgreSQL Backend

- **PostgreSQL**: 12+ with pgvector extension
- **RAM**: 8GB+ recommended
- **Disk**: Varies based on data size

## 🔧 Environment Setup

### PostgreSQL with pgvector

```bash
# Using Docker (recommended)
docker run -d \
  --name postgres-vectorstore \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vectorstore_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=vectorstore_db
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
```

### API Keys (if using external APIs)

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Cohere
export COHERE_API_KEY="your-cohere-api-key"

# Mistral AI
export MISTRAL_API_KEY="your-mistral-api-key"
```

### Django Settings (for PostgreSQL backend)

```python
import os
import django
from django.conf import settings

# Configure Django settings
settings.configure(
    INSTALLED_APPS=[
        'rakam_systems.ai_vectorstore.components.vectorstore',
    ],
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.getenv('POSTGRES_DB', 'vectorstore_db'),
            'USER': os.getenv('POSTGRES_USER', 'postgres'),
            'PASSWORD': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'HOST': os.getenv('POSTGRES_HOST', 'localhost'),
            'PORT': os.getenv('POSTGRES_PORT', '5432'),
        }
    }
)

django.setup()
```

## ✅ Verification

After installation, verify that the package is working:

```python
# Test core imports
from rakam_systems.ai_agents import BaseAgent, LLMGateway
from rakam_systems.ai_vectorstore import VectorStoreConfig, Node
from rakam_systems.ai_core import ConfigurationLoader, TrackingManager

print("✅ Installation successful!")
```

### Verify Specific Modules

```python
# Test AI Agents (if installed)
try:
    from rakam_systems.ai_agents import BaseAgent
    print("✅ AI Agents installed")
except ImportError:
    print("❌ AI Agents not installed")

# Test AI Vectorstore (if installed)
try:
    from rakam_systems.ai_vectorstore import VectorStoreConfig
    print("✅ AI Vectorstore installed")
except ImportError:
    print("❌ AI Vectorstore not installed")

# Test AI Core (always available)
from rakam_systems.ai_core import ConfigurationLoader
print("✅ AI Core installed")
```

### Django-dependent Components

Some components (like `ConfigurablePgVectorStore`) require Django to be configured:

```python
import django
from django.conf import settings

# Configure Django first
settings.configure(
    INSTALLED_APPS=['rakam_systems.ai_vectorstore'],
    DATABASES={...}
)
django.setup()

# Now you can import Django-dependent components
from rakam_systems.ai_vectorstore import ConfigurablePgVectorStore
print("✅ ConfigurablePgVectorStore available")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Module Not Found Error

```
ModuleNotFoundError: No module named 'rakam_systems'
```

**Solution**: Install from the correct directory:

```bash
cd app/rakam_systems
pip install -e .
```

#### 2. Missing Optional Dependencies

```
ImportError: cannot import name 'BaseAgent'
```

**Solution**: Install the required optional dependency group:

```bash
# For AI Agents
pip install -e ".[ai-agents]"

# For AI Vectorstore
pip install -e ".[ai-vectorstore]"
```

#### 3. Django Configuration Error

```
django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS, but settings are not configured.
```

**Solution**: Configure Django before importing Django-dependent components (see [Environment Setup](#environment-setup)).

#### 4. Torch Installation Issues

PyTorch can be large. Install CPU-only version for faster setup:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ai-vectorstore]"
```

#### 5. FAISS Import Error

For GPU support:

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

#### 6. Permission Errors

Use virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[all]"
```

### Getting Help

- **Documentation**: Check component-specific READMEs
- **Examples**: See `examples/` and `rakam_systems/examples/` directories
- **Issues**: [GitHub Issues](https://github.com/Rakam-AI/rakam_systems/issues)

## 🔄 Migration Guide

### Migrating from setup.cfg to pyproject.toml

If you were using the old `setup.cfg` based installation:

#### Step 1: Uninstall the old package

```bash
pip uninstall rakam-systems
```

#### Step 2: Install using pyproject.toml

```bash
cd app/rakam_systems
pip install -e ".[all]"
```

#### Step 3: Update your installation scripts

**Old way (deprecated):**

```bash
pip install -e .  # Installed ALL dependencies as required
```

**New way (recommended):**

```bash
# Install only what you need
pip install -e ".[ai-vectorstore]"  # Just vectorstore
pip install -e ".[ai-agents]"       # Just agents
pip install -e ".[all]"             # All features
pip install -e ".[complete]"        # Everything including dev
```

#### What Changed?

1. **Modular dependencies**: Dependencies are now optional and grouped by feature
2. **Core is minimal**: Base installation only includes 5 essential packages
3. **Explicit feature selection**: Choose which features to install
4. **Modern standards**: Follows PEP 621 and modern Python packaging

### Benefits of the New System

- ✅ **Faster installs**: Only install what you need
- ✅ **Smaller footprint**: Minimal core dependencies
- ✅ **Clear dependencies**: Easy to see what each feature requires
- ✅ **Better for CI/CD**: Install only required features in production
- ✅ **Standards compliant**: Uses modern Python packaging (PEP 621)

## 🚀 Development Installation

For contributing to Rakam Systems:

```bash
# Clone repository
git clone https://github.com/Rakam-AI/rakam_systems.git
cd rakam_systems/app/rakam_systems

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with development tools
pip install -e ".[complete]"

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📈 Upgrading

To upgrade an existing installation:

```bash
cd app/rakam_systems

# Pull latest changes
git pull

# Reinstall
pip install -e ".[all]" --upgrade
```

## 🗑️ Uninstallation

To uninstall:

```bash
pip uninstall rakam-systems
```

## 📖 Next Steps

After installation:

1. **Read the Quick Starts**:

   - [Main README](README.md)
   - [AI Vectorstore README](rakam_systems/ai_vectorstore/README.md)

2. **Explore Examples**:

   - `rakam_systems/examples/ai_vectorstore_examples/` - Vector store examples
   - `rakam_systems/examples/ai_agents_examples/` - Agent examples

3. **Check Documentation**:
   - Component-specific READMEs in each module
   - Example scripts with detailed comments

## 📝 Summary of Installation Commands

```bash
# Full installation (all features, no dev tools)
pip install -e ".[all]"

# Complete installation (everything including dev tools)
pip install -e ".[complete]"

# Core only (minimal)
pip install -e .

# AI Vectorstore only
pip install -e ".[ai-vectorstore]"

# AI Agents only
pip install -e ".[ai-agents]"

# Custom combination
pip install -e ".[ai-vectorstore,ai-agents,llm-gateway]"

# From locked requirements
pip install -r requirements.txt
```

---

**Note**: Always use a virtual environment to avoid conflicts with system packages.

**License**: Apache 2.0

**Support**: For questions and issues, visit our [GitHub repository](https://github.com/Rakam-AI/rakam_systems).

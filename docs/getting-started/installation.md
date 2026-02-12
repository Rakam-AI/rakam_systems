# Installation Guide

Complete installation instructions for Rakam Systems - a modular AI framework organized into three independent packages.

## 📑 Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Modular Installation](#modular-installation)
- [Installation Recipes](#installation-recipes)
- [Environment Setup](#environment-setup)
- [Verification](#verification)
- [Docker Setup](#docker-setup)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Option 1: Install All Packages

```bash
# Install all three packages
pip install -e ./rakam-system-core
pip install -e ./rakam-system-agent
pip install -e ./rakam-system-vectorstore

# Or in one command
pip install -e ./rakam-system-core ./rakam-system-agent ./rakam-system-vectorstore
```

### Option 2: Install Specific Packages

```bash
# Core only (required by others)
pip install -e ./rakam-system-core

# Agent package (includes core as dependency)
pip install -e ./rakam-system-agent

# Vectorstore package (includes core as dependency)
pip install -e ./rakam-system-vectorstore

# Agent + Vectorstore (for RAG applications)
pip install -e ./rakam-system-agent ./rakam-system-vectorstore
```

### Option 3: Install with Extras

```bash
# Vectorstore with specific backends
pip install -e "./rakam-system-vectorstore[postgres]"
pip install -e "./rakam-system-vectorstore[faiss]"
pip install -e "./rakam-system-vectorstore[all]"
```

---

## System Requirements

### Minimum Requirements

| Requirement    | Minimum               | Recommended  |
| -------------- | --------------------- | ------------ |
| **Python**     | 3.10                  | 3.11+        |
| **RAM**        | 4 GB                  | 8 GB+        |
| **Disk Space** | 1 GB                  | 5 GB+        |
| **OS**         | Linux, macOS, Windows | Linux, macOS |

### Module-Specific Requirements

#### AI Vectorstore (with local embeddings)

- **RAM**: 8 GB+ (embedding models are loaded into memory)
- **Disk**: 5 GB+ (for downloading model weights)
- **GPU**: Optional but recommended for faster inference
- **PostgreSQL**: 12+ with pgvector extension (for persistent storage)

#### AI Agents

- **RAM**: 4 GB minimum
- **Network**: Internet access for LLM API calls
- **API Keys**: OpenAI and/or Mistral API keys

---

## Modular Installation

Rakam Systems uses a modular architecture with three independent packages. Install only what you need:

### Core Package (Required)

The core package provides foundational interfaces and utilities. It's required by both agent and vectorstore packages.

```bash
pip install -e ./rakam-system-core
```

**Includes:**

- `pydantic` - Data validation
- `pyyaml` - YAML configuration
- `python-dotenv` - Environment variables
- `colorlog` - Logging
- `requests` - HTTP client

### Agent Package

The agent package provides AI agent implementations powered by Pydantic AI.

```bash
pip install -e ./rakam-system-agent
```

**Includes:**

- All core package dependencies
- `pydantic-ai` - Agent framework
- `mistralai` - Mistral AI provider
- `openai` - OpenAI provider
- `tiktoken` - Token counting

**Use Cases:**

- Building AI agents with tools
- Multi-step reasoning systems
- Conversational AI
- Structured output generation

### Vectorstore Package

The vectorstore package provides vector database solutions and document processing.

```bash
pip install -e ./rakam-system-vectorstore
```

**Includes:**

| Package                 | Purpose                          |
| ----------------------- | -------------------------------- |
| `sentence-transformers` | Local embedding models           |
| `huggingface-hub`       | HuggingFace model authentication |
| `faiss-cpu`             | Vector similarity search         |
| `psycopg2-binary`       | PostgreSQL driver                |
| `pgvector`              | PostgreSQL vector extension      |
| `django`                | ORM for database operations      |
| `torch`                 | Deep learning backend            |
| `pymupdf`               | PDF processing                   |
| `pymupdf4llm`           | Lightweight PDF to markdown      |
| `python-docx`           | DOCX processing                  |
| `beautifulsoup4`        | HTML parsing                     |
| `chonkie`               | Text chunking                    |
| `docling`               | Advanced document processing     |
| `openai`                | OpenAI embeddings (optional)     |
| `cohere`                | Cohere embeddings (optional)     |
| `odfpy`                 | ODT file processing              |
| `openpyxl`              | Excel file processing            |
| `pandas`                | Tabular data handling            |
| `joblib`                | Parallel processing              |

**Use Cases:**

- Semantic search applications
- RAG (Retrieval-Augmented Generation)
- Document Q&A systems
- Knowledge base management

### Package-Specific Extras

Each package supports optional extras for additional features:

```bash
# Vectorstore with PostgreSQL support
pip install -e "./rakam-system-vectorstore[postgres]"

# Vectorstore with FAISS support
pip install -e "./rakam-system-vectorstore[faiss]"

# Vectorstore with local embeddings
pip install -e "./rakam-system-vectorstore[local-embeddings]"

# Vectorstore with all features
pip install -e "./rakam-system-vectorstore[all]"
```

### Combining Packages

Install multiple packages for complete functionality:

```bash
# RAG application (agent + vectorstore)
pip install -e ./rakam-system-agent ./rakam-system-vectorstore

# Complete installation
pip install -e ./rakam-system-core ./rakam-system-agent "./rakam-system-vectorstore[all]"
```

---

## Installation Recipes

### Recipe 1: RAG Application

Build a Retrieval-Augmented Generation system:

```bash
pip install -e ./rakam-system-agent "./rakam-system-vectorstore[all]"
```

This gives you:

- Document loading and chunking
- Vector storage (FAISS or PostgreSQL)
- Embedding models
- AI agents for query processing

### Recipe 2: Simple Chatbot

Build a conversational AI without vector storage:

```bash
pip install -e ./rakam-system-agent
```

### Recipe 3: Document Processing Pipeline

Process and index documents without AI agents:

```bash
pip install -e "./rakam-system-vectorstore[all]"
```

### Recipe 4: Development Environment

Complete setup for contributing:

```bash
pip install -e ./rakam-system-core ./rakam-system-agent "./rakam-system-vectorstore[all]"
```

---

## Environment Setup

### API Keys

Create a `.env` file in your project root:

```bash
# OpenAI (for GPT models and embeddings)
OPENAI_API_KEY=sk-your-openai-key

# Mistral AI (for Mistral models)
MISTRAL_API_KEY=your-mistral-key

# Cohere (for Cohere embeddings)
COHERE_API_KEY=your-cohere-key

# HuggingFace (for private/gated models)
HUGGINGFACE_TOKEN=your-hf-token
```

Load in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

### PostgreSQL with pgvector

#### Option 1: Docker (Recommended)

```bash
docker run -d \
  --name postgres-vectorstore \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vectorstore_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

#### Option 2: Docker Compose

Create `docker-compose.yml`:

```yaml
version: "3.8"
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: vectorstore_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Run:

```bash
docker compose up -d
```

#### Environment Variables for PostgreSQL

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=vectorstore_db
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
```

### Django Configuration

For PostgreSQL-backed vector stores, configure Django:

```python
import os
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'rakam_system_vectorstore.components.vectorstore',
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
        },
        DEFAULT_AUTO_FIELD='django.db.models.BigAutoField',
    )
    django.setup()
```

---

## Verification

### Verify Core Installation

```python
# Test core imports
from rakam_system_core.ai_core.base import BaseComponent
from rakam_system_core.ai_core.interfaces import ToolComponent, VectorStore
print("✅ Core installed successfully!")
```

### Verify AI Agents

```python
try:
    from rakam_system_agent import BaseAgent
    from rakam_system_core.ai_core.interfaces.agent import AgentInput, AgentOutput
    print("✅ AI Agents installed successfully!")
except ImportError as e:
    print(f"❌ AI Agents not installed: {e}")
```

### Verify AI Vectorstore

```python
try:
    from rakam_system_vectorstore import (
        VectorStoreConfig,
        Node,
        VSFile,
        ConfigurableEmbeddings
    )
    print("✅ AI Vectorstore installed successfully!")
except ImportError as e:
    print(f"❌ AI Vectorstore not installed: {e}")
```

### Verify LLM Gateway

```python
try:
    from rakam_system_agent.components.llm_gateway import (
        OpenAIGateway,
        MistralGateway,
        LLMGatewayFactory
    )
    print("✅ LLM Gateway installed successfully!")
except ImportError as e:
    print(f"❌ LLM Gateway not installed: {e}")
```

### Full Verification Script

```python
#!/usr/bin/env python3
"""Verify rakam_systems installation."""

def check_module(name: str, import_fn):
    """Check if a module is properly installed."""
    try:
        import_fn()
        print(f"✅ {name}")
        return True
    except ImportError as e:
        print(f"❌ {name}: {e}")
        return False

def main():
    print("=" * 50)
    print("Rakam Systems Installation Verification")
    print("=" * 50)

    results = []

    # Core (always required)
    results.append(check_module(
        "Core",
        lambda: __import__('rakam_system_core.ai_core.base', fromlist=['BaseComponent'])
    ))

    # AI Agents
    results.append(check_module(
        "AI Agents",
        lambda: __import__('rakam_system_agent', fromlist=['BaseAgent'])
    ))

    # AI Vectorstore
    results.append(check_module(
        "AI Vectorstore",
        lambda: __import__('rakam_system_vectorstore', fromlist=['VectorStoreConfig'])
    ))

    # LLM Gateway
    results.append(check_module(
        "LLM Gateway",
        lambda: __import__('rakam_system_agent.components.llm_gateway', fromlist=['OpenAIGateway'])
    ))

    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} modules installed")

    if passed == total:
        print("🎉 All modules installed successfully!")
    else:
        print("⚠️  Some modules are missing. Install them with:")
        print('   pip install -e ".[all]"')

if __name__ == "__main__":
    main()
```

---

## Docker Setup

### Development with Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY app/rakam_systems /app/rakam_systems

# Install the package
WORKDIR /app/rakam_systems
RUN pip install -e ".[all]"

# Set working directory back
WORKDIR /app

# Default command
CMD ["python"]
```

### Docker Compose for Full Stack

```yaml
version: "3.8"

services:
  app:
    build: .
    volumes:
      - .:/app
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=vectorstore_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    depends_on:
      - postgres

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: vectorstore_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

```
ModuleNotFoundError: No module named 'rakam_systems'
```

**Solution:** Install from the correct directory:

```bash
cd app/rakam_systems
pip install -e .
```

#### 2. Missing Optional Dependencies

```
ImportError: cannot import name 'BaseAgent' from 'rakam_system_agent'
```

**Solution:** Install the required module:

```bash
pip install -e ".[ai-agents]"
```

#### 3. Django Not Configured

```
django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS...
```

**Solution:** Configure Django before importing Django-dependent components:

```python
import django
from django.conf import settings

settings.configure(
    INSTALLED_APPS=['rakam_system_vectorstore.components.vectorstore'],
    DATABASES={'default': {...}}
)
django.setup()

# Now import Django-dependent components
from rakam_system_vectorstore import ConfigurablePgVectorStore
```

#### 4. PyTorch Installation Issues

PyTorch is large (~2GB). For CPU-only installation:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ai-vectorstore]"
```

For CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[ai-vectorstore]"
```

#### 5. FAISS GPU Support

Replace CPU version with GPU version:

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

#### 6. libmagic Not Found

On macOS:

```bash
brew install libmagic
```

On Ubuntu/Debian:

```bash
apt-get install libmagic1
```

On Windows:

```bash
pip install python-magic-bin
```

#### 7. PostgreSQL Connection Refused

Ensure PostgreSQL is running:

```bash
# Check if running
docker ps | grep postgres

# Start if not running
docker start postgres-vectorstore

# Or start with docker compose
docker compose up -d postgres
```

#### 8. Permission Errors

Use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

pip install -e ".[all]"
```

### Getting Help

- **Documentation**: See `docs/` directory
- **Examples**: Check `rakam_systems/examples/`
- **Issues**: [GitHub Issues](https://github.com/Rakam-AI/rakam_systems/issues)

---

## Upgrading

### Upgrade to Latest Version

```bash
cd app/rakam_systems
git pull
pip install -e ".[all]" --upgrade
```

### Regenerate Locked Dependencies

```bash
pip install -e ".[complete]"
pip freeze > requirements.txt
```

---

## Uninstallation

```bash
pip uninstall rakam-systems
```

To remove all dependencies:

```bash
pip uninstall rakam-systems
pip freeze | xargs pip uninstall -y  # Be careful: removes ALL packages
```

---

## Next Steps

After installation:

1. **Read the Documentation**
   - [Components Guide](components.md)
   - [Development Guide](development_guide.md)

2. **Explore Examples**
   - `rakam_systems/examples/ai_agents_examples/`
   - `rakam_systems/examples/ai_vectorstore_examples/`

3. **Try Quick Start Code**

```python
import asyncio
from rakam_system_agent import BaseAgent

async def main():
    agent = BaseAgent(
        name="my_agent",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant."
    )
    result = await agent.arun("Hello, world!")
    print(result.output_text)

asyncio.run(main())
```

---

## Summary of Commands

```bash
# Core only
pip install -e .

# AI Agents
pip install -e ".[ai-agents]"

# AI Vectorstore
pip install -e ".[ai-vectorstore]"

# LLM Gateway
pip install -e ".[llm-gateway]"

# Document Loaders
pip install -e ".[loaders]"

# All features
pip install -e ".[all]"

# All features + dev tools
pip install -e ".[complete]"

# Custom combination
pip install -e ".[ai-agents,ai-vectorstore,dev]"
```

---

**License:** Apache 2.0

**Support:** [GitHub Repository](https://github.com/Rakam-AI/rakam_systems)

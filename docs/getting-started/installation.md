# Installation Guide

Complete installation instructions for Rakam Systems Components- a modular AI framework organized into three independent packages.

## 📑 Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Modular Installation](#modular-installation)
- [Environment Setup](#environment-setup)
- [Verification](#verification)
- [Docker Setup](#docker-setup)
- [Troubleshooting](#troubleshooting)

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

## Quick Start

### Option 1: Install All Packages

```bash
# Install all  components
pip install  rakam-systems==0.2.5rc5
```

### Option 2: Install Specific Packages

```bash
# Core only (required by others)
pip install  rakam-systems-core

# Agent package (includes core as dependency)
pip install rakam-systems-agent[all]==0.1.1rc9

# Vectorstore package (includes core as dependency)
pip install rakam-systems-vectorstore[all]==0.1.1rc10

# Agent + Vectorstore (for RAG applications)
pip install rakam-system-agent[all]==0.1.1rc9 rakam-systems-vectorstore[all]==0.1.1rc10
```

---

## Modular Installation

Rakam Systems uses a modular architecture with three independent packages. Install only what you need:

### Core Package (Required)

The core package provides foundational interfaces and utilities. It's required by both agent and vectorstore packages.

```bash
pip install rakam-systems-core
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
pip install rakam-systems-agent
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
pip install rakam-systems-vectorstore
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
            'rakam_systems_vectorstore.components.vectorstore',
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
from rakam_systems_core.base import BaseComponent
from rakam_systems_core.interfaces import ToolComponent, VectorStore
print("✅ Core installed successfully!")
```

### Verify AI Agents

```python
try:
    from rakam_systems_agent import BaseAgent
    from rakam_systems_core.interfaces.agent import AgentInput, AgentOutput
    print("✅ AI Agents installed successfully!")
except ImportError as e:
    print(f"❌ AI Agents not installed: {e}")
```

### Verify AI Vectorstore

```python
try:
    from rakam_systems_vectorstore import (
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
    from rakam_systems_agent.components.llm_gateway import (
        OpenAIGateway,
        MistralGateway,
        LLMGatewayFactory
    )
    print("✅ LLM Gateway installed successfully!")
except ImportError as e:
    print(f"❌ LLM Gateway not installed: {e}")
```

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
ImportError: cannot import name 'BaseAgent' from 'rakam_systems_agent'
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
    INSTALLED_APPS=['rakam_systems_vectorstore.components.vectorstore'],
    DATABASES={'default': {...}}
)
django.setup()

# Now import Django-dependent components
from rakam_systems_vectorstore import ConfigurablePgVectorStore
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
from rakam_systems_agent import BaseAgent

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

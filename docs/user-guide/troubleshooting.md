---
title: Troubleshooting
---

# Troubleshooting

## ModuleNotFoundError

```
ModuleNotFoundError: No module named 'rakam_systems'
```

Verify the package is installed:

```bash
pip install rakam-systems
```

## Missing optional dependencies

```
ImportError: cannot import name 'BaseAgent' from 'rakam_systems_agent'
```

Install the required package:

```bash
pip install rakam-systems-agent
```

## Django not configured

```
django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS...
```

Configure Django before importing Django-dependent components:

```python
import django
from django.conf import settings

settings.configure(
    INSTALLED_APPS=['rakam_systems_vectorstore.components.vectorstore'],
    DATABASES={'default': {...}}
)
django.setup()

from rakam_systems_vectorstore import ConfigurablePgVectorStore
```

## PyTorch installation issues

PyTorch is large (~2 GB). For CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## FAISS GPU support

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

## libmagic not found

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

## PostgreSQL connection refused

```bash
# Check if running
docker ps | grep postgres

# Start if not running
docker start postgres-vectorstore

# Or start with docker compose
docker compose up -d postgres
```

## Verify your installation

```python
# Core
from rakam_systems_core.base import BaseComponent
from rakam_systems_core.interfaces import ToolComponent, VectorStore
print("✅ Core installed successfully!")
```

```python
# Agent
try:
    from rakam_systems_agent import BaseAgent
    from rakam_systems_core.interfaces.agent import AgentInput, AgentOutput
    print("✅ AI Agents installed successfully!")
except ImportError as e:
    print(f"❌ AI Agents not installed: {e}")
```

```python
# Vectorstore
try:
    from rakam_systems_vectorstore import VectorStoreConfig, Node, VSFile, ConfigurableEmbeddings
    print("✅ AI Vectorstore installed successfully!")
except ImportError as e:
    print(f"❌ AI Vectorstore not installed: {e}")
```

```python
# LLM Gateway
try:
    from rakam_systems_agent.components.llm_gateway import OpenAIGateway, MistralGateway, LLMGatewayFactory
    print("✅ LLM Gateway installed successfully!")
except ImportError as e:
    print(f"❌ LLM Gateway not installed: {e}")
```

## Get help

- **Issues:** [GitHub Issues](https://github.com/Rakam-AI/rakam_systems/issues)

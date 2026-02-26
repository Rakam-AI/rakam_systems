---
title: Getting Started
---

# Getting Started

import Prerequisites from '../_partials/_prerequisites.md';

<Prerequisites />

## Installation

### Install all packages

Install all Rakam Systems packages at once:

```bash
pip install rakam-systems
```

This is the recommended starting point. It includes the core, agent, vectorstore, tools, and CLI packages.

### Install specific packages

Rakam Systems uses a modular architecture. If you want to reduce dependencies, you can install only what you need:

```bash
# Core only (required by all other packages)
pip install rakam-systems-core

# Agent package (includes core)
pip install rakam-systems-agent[all]

# Vectorstore package (includes core and tools)
pip install rakam-systems-vectorstore[all]

# Tools package (evaluation, S3 utilities)
pip install rakam-systems-tools

# CLI
pip install rakam-systems-cli

# Agent + Vectorstore (for RAG applications)
pip install rakam-systems-agent[all] rakam-systems-vectorstore[all]
```

### Dependencies

| Package | Purpose | Key dependencies |
|---------|---------|-----------------|
| `rakam-systems-core` | Foundational interfaces and utilities. Required by all other packages. | `pydantic`, `PyYAML` |
| `rakam-systems-agent` | AI agent framework powered by Pydantic AI. `[all]` adds LLM provider clients. | Core + `pydantic-ai`, `python-dotenv`; `[all]`: `openai`, `mistralai` |
| `rakam-systems-vectorstore` | Vector storage and document processing. `[all]` adds all backends and loaders. | Core + Tools + `numpy`; `[all]`: `faiss-cpu`, `sentence-transformers`, `torch`, `pgvector`, `openai`, `cohere` |
| `rakam-systems-tools` | Evaluation framework and S3 utilities. | `pydantic`, `boto3`, `requests` |
| `rakam-systems-cli` | Command-line interface (`rakam` command). | Tools + `typer` |


## Environment Setup

### Configure API keys

Create a `.env` file in your project root. Not all keys are required — they depend on which providers and features you use.

```bash
# OpenAI — required for GPT models and OpenAI embeddings
OPENAI_API_KEY=sk-your-openai-key

# Mistral AI — required for Mistral models
MISTRAL_API_KEY=your-mistral-key

# Cohere — required for Cohere embeddings
COHERE_API_KEY=your-cohere-key

# HuggingFace — required for private or gated models
HUGGINGFACE_TOKEN=your-hf-token
```

Load in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Configure PostgreSQL

PostgreSQL with pgvector is required if you use `ConfigurablePgVectorStore` or PostgreSQL-backed chat history. This guide assumes a PostgreSQL instance with pgvector is already available.

Configure the connection via environment variables:

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vectorstore_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

---
title: User Guide
---

import Prerequisites from '../_partials/_prerequisites.md';

This guide is for AI engineers using Rakam Systems to build AI systems. It covers installation, environment setup, agents, vector stores, evaluation, and cloud storage.

<Prerequisites />

## Installation

### Install all packages

Install all Rakam Systems packages at once:

```bash
pip install rakam-systems
```

This is the recommended starting point. It includes the core, agent, and vectorstore packages.

### Install specific packages

Rakam Systems uses a modular architecture. If you want to reduce dependencies, you can install only what you need:

```bash
# Core only (required by all other packages)
pip install rakam-systems-core

# Agent package (includes core)
pip install rakam-systems-agent[all]

# Vectorstore package (includes core)
pip install rakam-systems-vectorstore[all]

# Agent + Vectorstore (for RAG applications)
pip install rakam-systems-agent[all] rakam-systems-vectorstore[all]
```

### Review dependencies

| Package | Purpose | Min specs | Key dependencies |
|---------|---------|-----------|-----------------|
| `rakam-systems-core` | Foundational interfaces and utilities. Required by all other packages. | — | `pydantic`, `pyyaml`, `python-dotenv`, … |
| `rakam-systems-agent` | AI agent implementations powered by Pydantic AI. | 4 GB RAM, internet access | Core + `pydantic-ai`, `mistralai`, `openai`, … |
| `rakam-systems-vectorstore` | Vector storage and document processing. Requires PostgreSQL + pgvector for persistent storage. | 8 GB+ RAM, 5 GB+ disk | Core + `sentence-transformers`, `faiss-cpu`, `torch`, … |


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

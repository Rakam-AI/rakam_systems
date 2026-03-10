# Rakam Systems

**Rakam Systems** is a platform designed to industrialize the construction, deployment, and operation of enterprise-grade AI systems with a focus on quality, scalability, and production-readiness.

## Overview

Rakam Systems was born from an internal need at Rakam AI. For every new AI project, teams faced recurring technical challenges: collecting test data, evaluating quality, orchestrating components, configuring cloud infrastructure, and ensuring regulatory compliance. Rather than rebuilding these elements each time, Rakam decided to standardize and automate the entire AI production pipeline.

## Why Rakam Systems

- **State-of-the-Art Technology**: FastAPI, Pydantic AI, FAISS, pgvector, Sentence Transformers, OpenAI, Mistral AI
- **Production-First**: Type safety, structured data exchange, scalable architecture, Docker templates
- **Open Source**: Transparent design, community-driven, standard tooling

## Core Components

Rakam Systems provides modular, independently installable packages:

| Package | Description |
|---------|-------------|
| `rakam-systems-core` | Foundational interfaces and utilities required by all other packages |
| `rakam-systems-agent` | AI agent implementations with multi-LLM support and tool integration |
| `rakam-systems-vectorstore` | Vector storage and document processing for semantic search and RAG |
| `rakam-systems-tools` | Evaluation tools, cloud storage utilities, and monitoring |
| `rakam-systems-cli` | Command-line interface for running evaluations and tracking quality |


## Installation

### Install all packages

```bash
pip install rakam-systems
```

### Install specific packages

```bash
# Core only (required by all other packages)
pip install rakam-systems-core

# Agent package
pip install rakam-systems-agent[all]

# Vectorstore package
pip install rakam-systems-vectorstore[all]

# Tools package (evaluation, S3 utilities)
pip install rakam-systems-tools

# CLI
pip install rakam-systems-cli

# Agent + Vectorstore (for RAG applications)
pip install rakam-systems-agent[all] rakam-systems-vectorstore[all]
```

### Requirements

- Python 3.10 or higher (3.8+ for core, tools, and cli)

## Use Cases

With Rakam Systems, you can build:

- **Retrieval-Augmented Generation (RAG) Systems**: Combine vector retrieval with LLM prompt generation
- **Agent Systems**: Create modular agents that perform specific tasks using LLMs
- **Chained Gen AI Systems**: Chain multiple AI tasks for complex workflows
- **Search Engines**: Semantic search over documents using fine-tuned embeddings
- **Any Custom AI System**: Use components to create any AI solution tailored to your needs

## Documentation

Full documentation is available in the [`docs/`](./docs/) directory:

- [Introduction](./docs/introduction.md)
- [Getting Started](./docs/getting-started.md)
- [User Guide](./docs/user-guide/index.md)
- [Developer Guide](./docs/developer-guide/index.md)

## Contributing

We welcome contributions! To contribute:

1. Fork the repository and clone it locally.
2. Create a feature branch: `git checkout -b feature-branch`
3. Install the package(s) you are working on:
   ```bash
   cd <package-dir>
   uv sync --all-extras --dev
   ```
4. Make your changes and run tests: `uv run pytest -v`
5. Commit with a meaningful message and submit a pull request.

For more details, see [Contributing](./docs/contributing/index.md).

## License

This project is licensed under the Apache-2.0 license.

## Support

For any issues, questions, or suggestions, please contact [mohammed@rakam.ai](mailto:mohammed@rakam.ai) or open an issue on GitHub.

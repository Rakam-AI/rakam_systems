---
title: Developer Guide
---

# Developer Guide

This guide is for developers who want to understand the internals of Rakam Systems and contribute to the project. It covers the architecture, package structure, interfaces, and design patterns. For usage patterns and tutorials, see the [User Guide](../user-guide/index.md).

## Set up the development environment

```bash
git clone git@github.com:Rakam-AI/rakam_systems.git
cd rakam_systems
python3 -m venv venv
source venv/bin/activate
pip install -e core/ -e ai-components/agents/ -e ai-components/vector-store/ -e tools/ -e cli/
```

## Architecture Overview

Rakam Systems is organized into five packages:

```
rakam-systems/
│
├─ core/                              ← Foundation (required by all others)
│  └─ src/rakam_systems_core/
│     ├─ base.py                         BaseComponent
│     ├─ interfaces/                     Abstract interfaces
│     ├─ config_loader.py                Configuration system
│     ├─ tracking.py                     Input/output tracking
│     └─ utils/                          Logging utilities
│
├─ ai-components/
│  ├─ agents/                         ← Agent implementations
│  │  └─ src/rakam_systems_agent/
│  │     └─ components/
│  │        ├─ base_agent.py             BaseAgent implementation
│  │        ├─ llm_gateway/              LLM provider gateways
│  │        ├─ chat_history/             Chat history backends
│  │        └─ tools/                    Built-in tools
│  │
│  └─ vector-store/                   ← Vector storage and document processing
│     └─ src/rakam_systems_vectorstore/
│        ├─ core.py                      Node, VSFile data structures
│        ├─ config.py                    VectorStoreConfig
│        └─ components/
│           ├─ vectorstore/              Store implementations
│           ├─ embedding_model/          Embedding models
│           ├─ loader/                   Document loaders
│           └─ chunker/                  Text chunkers
│
├─ tools/                             ← Evaluation SDK, S3 utilities, observability
│  └─ src/rakam_systems_tools/
│     ├─ evaluation/                     DeepEvalClient, metrics, schemas
│     └─ utils/
│        ├─ s3/                          S3-compatible storage wrapper
│        ├─ logging.py                   Logging utilities
│        ├─ metrics.py                   Metrics collection
│        └─ tracing.py                   Tracing utilities
│
└─ cli/                               ← Command-line interface for evaluations
   └─ src/rakam_systems_cli/
      ├─ cli.py                          CLI entry point (rakam eval ...)
      ├─ decorators.py                   @eval_run decorator
      └─ utils/                          CLI utilities
```

### Design Principles

- **Modular Architecture**: Five packages that can be installed separately
- **Clear Dependencies**: Agent, vectorstore, tools, and CLI packages depend on core
- **Component-Based**: All components extend `BaseComponent` with lifecycle management (`setup()`, `shutdown()`)
- **Interface-Driven**: Abstract interfaces define contracts for extensibility
- **Configuration-First**: YAML/JSON configuration support for all components
- **Provider-Agnostic**: Support for multiple LLM providers, embedding models, and vector stores

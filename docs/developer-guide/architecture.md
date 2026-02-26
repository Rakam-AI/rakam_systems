---
title: Architecture
---

# Architecture

Rakam Systems is organized into five packages. See the [source code layout](./getting-started.md#source-code-architecture) for the full directory tree.

## Design principles

- **Modular architecture**: Five packages that can be installed separately
- **Clear dependencies**: Agent, vectorstore, tools, and CLI packages depend on core
- **Component-based**: All components extend `BaseComponent` with lifecycle management (`setup()`, `shutdown()`)
- **Interface-driven**: Abstract interfaces define contracts for extensibility
- **Configuration-first**: YAML/JSON configuration support for all components
- **Provider-agnostic**: Support for multiple LLM providers, embedding models, and vector stores

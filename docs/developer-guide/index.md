---
title: Developer Guide
---

# Developer Guide

This guide is for developers who want to understand the internals of Rakam Systems and build with its full API surface. It covers architecture, core abstractions, agents, vector pipelines, search, configuration, and tracking. For usage patterns and tutorials, see the [User Guide](../user-guide/index.md).

## Set up the development environment

```bash
git clone git@github.com:Rakam-AI/rakam_systems.git
cd rakam_systems
python3 -m venv venv
source venv/bin/activate
pip install -e core/ -e ai-components/agents/ -e ai-components/vector-store/ -e tools/ -e cli/
```

## Sections

1. [Architecture](./architecture.md) — Package structure and design principles
2. [Core concepts](./core.md) — BaseComponent lifecycle, interfaces, configuration-first design
3. [Build agents](./agents.md) — BaseAgent, tools, ToolRegistry, dynamic system prompts, structured output
4. [Use LLM gateways](./llm-gateways.md) — OpenAI, Mistral, Factory; direct generation, structured output, streaming
5. [Manage chat history](./chat-history.md) — JSON, SQLite, PostgreSQL backends; Pydantic AI integration
6. [Build vector pipelines](./vectorstore.md) — Data structures, embeddings, document loading, chunking
7. [Search](./search.md) — Vector search, keyword search, hybrid search
8. [Configure with YAML](./configuration.md) — Agent configuration, VectorStore configuration, multi-environment deployment
9. [Track and evaluate](./tracking.md) — Tracking system, cross-reference evaluation CLI and SDK
10. [Environment variables](./environment.md) — Complete reference table

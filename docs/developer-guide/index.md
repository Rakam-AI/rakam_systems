---
title: Developer Guide
---

# Developer Guide

This guide is for developers who want to understand the internals of Rakam Systems and build with its full API surface. It covers architecture, core abstractions, agents, vector pipelines, search, configuration, and tracking. For usage patterns and tutorials, see the [User Guide](../user-guide/index.md).

## Sections

1. [Getting Started](./getting-started.md) — Set up the development environment
2. [Architecture](./architecture.md) — Package structure and design principles
3. [Core concepts](./core.md) — BaseComponent lifecycle, interfaces, configuration-first design
4. [Build agents](./agents.md) — BaseAgent, tools, ToolRegistry, dynamic system prompts, structured output
5. [Use LLM gateways](./llm-gateways.md) — OpenAI, Mistral, Factory; direct generation, structured output, streaming
6. [Manage chat history](./chat-history.md) — JSON, SQLite, PostgreSQL backends; Pydantic AI integration
7. [Build vector pipelines](./vectorstore.md) — Data structures, embeddings, document loading, chunking
8. [Search](./search.md) — Vector search, keyword search, hybrid search
9. [Configure with YAML](./configuration.md) — Agent configuration, VectorStore configuration, multi-environment deployment
10. [Track and evaluate](./tracking.md) — Tracking system, cross-reference evaluation CLI and SDK
11. [Environment variables](./environment.md) — Complete reference table

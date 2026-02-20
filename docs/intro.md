---
title: Introduction
---

# Introduction

**Rakam Systems** is a platform designed to industrialize the construction, deployment, and operation of enterprise-grade AI systems with a focus on quality, scalability, and production-readiness.

## Origins

Rakam Systems was born from an internal need at Rakam AI. For every new AI project, teams faced recurring technical challenges: collecting test data, evaluating quality, orchestrating components, configuring cloud infrastructure, and ensuring regulatory compliance. Rather than rebuilding these elements each time, Rakam decided to standardize and automate the entire AI production pipeline.

This platform represents years of expertise in building production AI systems, distilled into a comprehensive framework that follows industry best practices and leverages state-of-the-art technologies.

## Target Users

Rakam Systems is built for **AI teams** working on production systems, serving three primary personas:

- **AI Engineers**: who build and maintain AI systems
- **Technical Leads**: who architect and guide AI development
- **Platform Teams**: who need visibility into system performance, quality metrics, and production readiness

## Why Rakam Systems

Rakam Systems was built to solve real production challenges encountered while deploying demanding AI systems. The framework reflects years of experience in what actually works at scale, distilled into reusable components and patterns.

**State-of-the-Art Technology**

We select proven, cutting-edge technologies for each layer of the stack:

- **Modern Python frameworks**: FastAPI for APIs, Pydantic for data validation and type safety
- **Leading AI libraries**: Pydantic AI for agent orchestration, integration with OpenAI, Anthropic, and Mistral
- **Production-grade vector search**: FAISS for similarity search, PostgreSQL with pgvector for persistent storage
- **Best-in-class ML tools**: Sentence Transformers and Hugging Face models for embeddings, PyTorch as the deep learning backend
- **Robust data processing**: BeautifulSoup4 for HTML, PyMuPDF for PDFs, pandas for tabular data

The platform stays current through continuous updates, ensuring access to the latest advancements while maintaining stability for production deployments.

**Production-First Framework**

Every design decision prioritizes real-world deployment:

- **Type safety throughout**: Pydantic models ensure data consistency across component boundaries
- **Structured data exchange**: Components communicate via well-defined schemas, making integration predictable
- **Scalable architecture**: Libraries chosen for efficiency (FAISS for vector operations, Django ORM for database access)
- **Production templates included**: FastAPI service templates with Docker configurations for immediate deployment
- **Observability built-in**: Evaluation framework for quality monitoring, metrics collection, and compliance tracking

The framework emerged from actual production requirements—performance bottlenecks, debugging challenges, scaling issues—and addresses them systematically.

**Open Source**

Rakam Systems is fully open source and welcomes contributions:

- **Transparent design**: All architectural decisions are documented and open for discussion
- **Community-driven**: We welcome improvements, new integrations, and feature proposals
- **Standard tooling**: Built on widely-adopted open source libraries (torch, Django, PostgreSQL)
- **Clear contribution paths**: Issues, pull requests, and discussions are encouraged

The codebase is public because we believe the best frameworks emerge from collective expertise and real-world testing.

## Core Components

Rakam Systems provides modular, independently installable packages:

- **Core** - Foundational interfaces and utilities required by all other packages
- **Agents** - AI agent implementations with multi-LLM support and tool integration
- **Vector Store** - Vector storage and document processing for semantic search and RAG applications
- **Tools** - Evaluation framework, cloud storage utilities, and monitoring capabilities
- **CLI** - Command-line interface for running evaluations and tracking quality

## Supporting Services & Tools

Managed separately with independent versioning:

- **Evaluation Services** - Deployed as standalone services. Available as a public Docker image for external users.
- **Templates** - Project templates (e.g., FastAPI microservice template) that provide starting points for new AI systems using Rakam Systems components.

---

- **GitHub Repo:** https://github.com/Rakam-AI/rakam_systems
- **Create an Issue:** https://github.com/Rakam-AI/rakam_systems/issues/new/choose

:::note
Use the version dropdown (top-right) to switch between versions.
:::

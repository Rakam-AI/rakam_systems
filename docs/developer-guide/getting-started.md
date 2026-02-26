---
title: Getting Started
---

# Getting Started

import Prerequisites from '../_partials/_prerequisites.md';

<Prerequisites />

## Clone the repository

```bash
git clone git@github.com:Rakam-AI/rakam_systems.git
cd rakam_systems
```

## Source code architecture

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

## Set up the development environment

Create a virtual environment and install all packages in editable mode (source changes take effect immediately without reinstalling):

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows: venv\Scripts\activate

pip install -e core/ -e ai-components/agents/[all] -e ai-components/vector-store/[all] -e tools/ -e cli/
```

Verify the installation:

```bash
python -c "from rakam_systems_agent import BaseAgent; print('OK')"
rakam --help
```

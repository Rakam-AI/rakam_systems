# Vector Store MCP Server

This module provides an MCP (Model Context Protocol) server for vector store operations, enabling AI agents to interact with vector databases through a standardized interface.

## Overview

The Vector Store MCP Server wraps vector store operations into reusable tool components. It allows AI agents to:

- Search for documents using semantic similarity
- Add documents to the vector store
- Query collection information and metadata
- Perform concurrent operations efficiently

## Components

| Component | Purpose | Key Parameters |
|-----------|---------|---------------|
| **VectorSearchTool** | Semantic search on collections | `query`, `collection_name`, `top_k` |
| **VectorStorageTool** | Add documents to the store | `documents`, `collection_name`, `doc_metadata` |
| **VectorInfoTool** | Retrieve collection info | `collection_name` (optional) |

## Quick Start

```python
from rakam_systems_vectorstore.server import run_vector_mcp
from rakam_systems_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore
from rakam_systems_vectorstore.config import VectorStoreConfig, EmbeddingConfig

config = VectorStoreConfig(
    name="my_store",
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",
        model_name="Snowflake/snowflake-arctic-embed-m"
    )
)

vector_store = ConfigurablePgVectorStore(name="my_store", config=config)
vector_store.setup()

mcp_server = run_vector_mcp(vector_store)
print(mcp_server.list_components())
# Output: ['vector_info', 'vector_search', 'vector_storage']
```

## Sending Messages

```python
# Search for documents
result = await mcp_server.asend_message(
    sender="agent",
    receiver="vector_search",
    message={'arguments': {'query': 'machine learning', 'top_k': 5}}
)

# Add documents
result = await mcp_server.asend_message(
    sender="agent",
    receiver="vector_storage",
    message={'arguments': {'documents': ['Doc 1', 'Doc 2'], 'collection_name': 'docs'}}
)

# Get collection info
result = await mcp_server.asend_message(
    sender="agent",
    receiver="vector_info",
    message={'arguments': {'collection_name': 'docs'}}
)
```

## Function Reference

### run_vector_mcp()

```python
def run_vector_mcp(
    vector_store,
    name: str = "vector_store_mcp",
    enable_logging: bool = False
) -> MCPServer
```

**Parameters:**

- `vector_store`: ConfigurablePgVectorStore instance (must be set up)
- `name` (str, optional): Name for the MCP server (default: "vector_store_mcp")
- `enable_logging` (bool, optional): Enable detailed MCP logging (default: False)

**Returns:** `MCPServer` — configured instance with registered tools

## Architecture

```
┌─────────────────────────────────────────┐
│          AI Agent / Client              │
└──────────────┬──────────────────────────┘
               │ send_message / asend_message
               ▼
┌─────────────────────────────────────────┐
│         MCPServer                       │
│  (Message Router & Component Registry)  │
└──────────────┬──────────────────────────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
┌─────────┐ ┌──────────┐ ┌──────────┐
│ Vector  │ │ Vector   │ │ Vector   │
│ Search  │ │ Storage  │ │ Info     │
└────┬────┘ └────┬─────┘ └────┬─────┘
     └───────────┴─────────────┘
                 │
                 ▼
     ┌────────────────────────┐
     │  ConfigurablePgVector  │
     │       Store            │
     └────────────────────────┘
```

## Related Documentation

- [MCP Server API](../../../../../core/src/rakam_systems_core/mcp/README.md)
- [Agent MCP Server](../../../../agents/src/rakam_systems_agent/server/README.md)
- [Official documentation](https://rakam-ai.github.io/rakam-systems-docs/)

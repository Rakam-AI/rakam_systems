# Vector Store MCP Server

This module provides an MCP (Model Context Protocol) server for vector store operations, enabling AI agents to interact with vector databases through a standardized interface.

## Overview

The Vector Store MCP Server creates a message-based interface that wraps vector store operations into reusable tool components. This allows AI agents to:

- Search for documents using semantic similarity
- Add documents to the vector store
- Query collection information and metadata
- Perform concurrent operations efficiently

## Components

### 1. VectorSearchTool

Performs semantic search on vector collections.

**Parameters:**
- `query` (str): Search query text
- `collection_name` (str, optional): Target collection (default: "documents")
- `top_k` (int, optional): Number of results to return (default: 5)

**Returns:**
```python
{
    'query': str,
    'collection': str,
    'results_count': int,
    'results': List[{
        'content': str,
        'node_id': str,
        'source_file': str,
        'position': int,
        'metadata': dict
    }]
}
```

### 2. VectorStorageTool

Adds documents to the vector store.

**Parameters:**
- `documents` (List[str]): List of document texts to add
- `collection_name` (str, optional): Target collection (default: "documents")
- `doc_metadata` (dict, optional): Metadata to attach to documents

**Returns:**
```python
{
    'success': bool,
    'collection': str,
    'documents_added': int,
    'node_ids': List[str]
}
```

### 3. VectorInfoTool

Retrieves information about collections.

**Parameters:**
- `collection_name` (str, optional): Specific collection name (lists all if not provided)

**Returns:**
```python
# For specific collection:
{
    'collection_name': str,
    'node_count': int,
    'embedding_dim': int
}

# For all collections:
{
    'total_collections': int,
    'collections': List[str]
}
```

## Usage

### Basic Setup

```python
from rakam_systems.ai_vectorstore.server import run_vector_mcp
from rakam_systems.ai_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore
from rakam_systems.ai_vectorstore.config import VectorStoreConfig, EmbeddingConfig

# Initialize vector store
config = VectorStoreConfig(
    name="my_store",
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",
        model_name="Snowflake/snowflake-arctic-embed-m"
    )
)

vector_store = ConfigurablePgVectorStore(name="my_store", config=config)
vector_store.setup()

# Create MCP server
mcp_server = run_vector_mcp(vector_store)

# List available tools
print(mcp_server.list_components())
# Output: ['vector_info', 'vector_search', 'vector_storage']
```

### Using with Async Messages

```python
# Search for documents
result = await mcp_server.asend_message(
    sender="agent",
    receiver="vector_search",
    message={
        'arguments': {
            'query': 'machine learning',
            'collection_name': 'docs',
            'top_k': 5
        }
    }
)

# Add documents
result = await mcp_server.asend_message(
    sender="agent",
    receiver="vector_storage",
    message={
        'arguments': {
            'documents': ['Doc 1 text', 'Doc 2 text'],
            'collection_name': 'docs'
        }
    }
)

# Get collection info
result = await mcp_server.asend_message(
    sender="agent",
    receiver="vector_info",
    message={
        'arguments': {
            'collection_name': 'docs'
        }
    }
)
```

### Integration with Tool Registry

```python
from rakam_systems.ai_core.interfaces import ToolRegistry, ToolInvoker

# Create registry and invoker
registry = ToolRegistry()
invoker = ToolInvoker(registry)

# Register MCP server
invoker.register_mcp_server("vector_store_mcp", mcp_server)

# Register tools
registry.register_mcp_tool(
    name="search_docs",
    mcp_server="vector_store_mcp",
    mcp_tool_name="vector_search",
    description="Search documents using semantic similarity",
    category="vector_store"
)

# Use through invoker
results = await invoker.ainvoke(
    "search_docs",
    query="What is AI?",
    top_k=3
)
```

### Concurrent Operations

```python
import asyncio

# Execute multiple operations concurrently
results = await asyncio.gather(
    mcp_server.asend_message(
        sender="agent",
        receiver="vector_search",
        message={'arguments': {'query': 'machine learning', 'top_k': 3}}
    ),
    mcp_server.asend_message(
        sender="agent",
        receiver="vector_search",
        message={'arguments': {'query': 'deep learning', 'top_k': 3}}
    ),
    mcp_server.asend_message(
        sender="agent",
        receiver="vector_info",
        message={'arguments': {}}
    )
)
```

## Function Reference

### run_vector_mcp()

Creates and configures an MCP server with vector store tools.

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

**Returns:**
- `MCPServer`: Configured MCP server instance with registered tools

## Examples

See the comprehensive examples in:
- `/examples/mcp_vector_search_example.py` - Full MCP vector search demonstration
- `/docs/MCP_VECTOR_STORE_GUIDE.md` - Complete guide to MCP with vector stores

## Architecture

The MCP server follows a modular architecture:

```
┌─────────────────────────────────────────┐
│          AI Agent / Client              │
└──────────────┬──────────────────────────┘
               │
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
│ Tool    │ │ Tool     │ │ Tool     │
└────┬────┘ └────┬─────┘ └────┬─────┘
     │           │             │
     └───────────┴─────────────┘
                 │
                 ▼
     ┌────────────────────────┐
     │  ConfigurablePgVector  │
     │       Store            │
     └────────────────────────┘
```

## Benefits

1. **Decoupled Architecture**: Tools don't need direct dependencies on the vector store
2. **Standardized Interface**: All tools work the same way through MCP
3. **Easy Extension**: Add new tools without modifying existing code
4. **Tool Discovery**: Agents can query available tools dynamically
5. **Async Support**: Efficient concurrent operations
6. **Integration Ready**: Works seamlessly with BaseAgent and other AI components

## Related Documentation

- [MCP Server API](/app/rakam_systems/rakam_systems/ai_core/mcp/README.md)
- [MCP Vector Store Guide](/docs/MCP_VECTOR_STORE_GUIDE.md)
- [MCP Quickstart](/docs/MCP_QUICKSTART.md)
- [Vector Search Architecture](/docs/VECTOR_SEARCH_ARCHITECTURE.md)


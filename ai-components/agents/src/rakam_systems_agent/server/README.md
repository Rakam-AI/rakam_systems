# Agent MCP Server

This module provides an MCP (Model Context Protocol) server for AI agent components, enabling flexible registration and communication between agent tools and utilities through a standardized message-based interface.

## Overview

The Agent MCP Server creates a message-based component registry that allows AI agents to dynamically register, discover, and invoke tools. This provides:

- **Flexible component registration**: Add any BaseComponent implementation
- **Unified communication**: All components communicate through the same interface
- **Dynamic tool discovery**: Agents can query available tools at runtime
- **Async support**: Efficient concurrent operations
- **Decoupled architecture**: Tools don't need to know about each other

## Quick Start

```python
from rakam_systems_agent.server import run_agent_mcp
from rakam_systems_core.base import BaseComponent

class SearchTool(BaseComponent):
    def run(self, query: str, max_results: int = 10):
        return f"Found {max_results} results for: {query}"

class CalculatorTool(BaseComponent):
    def run(self, operation: str, a: float, b: float):
        if operation == "add":
            return a + b
        return None

search = SearchTool(name="search")
calculator = CalculatorTool(name="calculator")

mcp_server = run_agent_mcp([search, calculator])
print(mcp_server.list_components())
# Output: ['calculator', 'search']
```

## Sending Messages

```python
# Synchronous
result = mcp_server.send_message(
    sender="agent",
    receiver="search",
    message={'arguments': {'query': 'machine learning', 'max_results': 5}}
)

# Asynchronous
result = await mcp_server.asend_message(
    sender="agent",
    receiver="search",
    message={'arguments': {'query': 'deep learning', 'max_results': 3}}
)
```

## Function Reference

### run_agent_mcp()

```python
def run_agent_mcp(
    components: Iterable[BaseComponent],
    name: str = "agent_mcp",
    enable_logging: bool = False
) -> MCPServer
```

**Parameters:**

- `components` (Iterable[BaseComponent]): Components to register with the server
- `name` (str, optional): Name for the MCP server (default: "agent_mcp")
- `enable_logging` (bool, optional): Enable detailed MCP logging (default: False)

**Returns:** `MCPServer` — configured instance with registered components

## Component Requirements

Components must:

1. Inherit from `BaseComponent`
2. Have a unique name
3. Implement `run(*args, **kwargs)` or `handle_message(sender, message)`

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
     ┌─────────┼─────────┬─────────┐
     ▼         ▼         ▼         ▼
┌─────────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ Search  │ │Calc  │ │ DB   │ │ ...  │
│ Tool    │ │Tool  │ │Tool  │ │      │
└─────────┘ └──────┘ └──────┘ └──────┘
  (BaseComponent implementations)
```

## Comparison with Vector Store MCP Server

| Feature            | Agent MCP                   | Vector Store MCP                  |
| ------------------ | --------------------------- | --------------------------------- |
| **Purpose**        | General agent components    | Vector store operations           |
| **Components**     | User-provided               | Pre-built (search, storage, info) |
| **Registration**   | Dynamic, flexible           | Automatic with vector store       |
| **Use Case**       | Custom tools, utilities     | Document search, RAG systems      |
| **Initialization** | `run_agent_mcp(components)` | `run_vector_mcp(vector_store)`    |

Both servers use the same underlying `MCPServer` and can be used together in a single system.

## Related Documentation

- [MCP Server API](../../../../../core/src/rakam_systems_core/mcp/README.md)
- [Vector Store MCP Server](../../../../vector-store/src/rakam_systems_vectorstore/server/README.md)
- [BaseComponent API](../../../../../core/src/rakam_systems_core/base.py)
- [Official documentation](https://rakam-ai.github.io/rakam-systems-docs/)

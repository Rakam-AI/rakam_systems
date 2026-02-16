# Agent MCP Server

This module provides an MCP (Model Context Protocol) server for AI agent components, enabling flexible registration and communication between agent tools and utilities through a standardized message-based interface.

## Overview

The Agent MCP Server creates a message-based component registry that allows AI agents to dynamically register, discover, and invoke tools. This provides:

- **Flexible component registration**: Add any BaseComponent implementation
- **Unified communication**: All components communicate through the same interface
- **Dynamic tool discovery**: Agents can query available tools at runtime
- **Async support**: Efficient concurrent operations
- **Decoupled architecture**: Tools don't need to know about each other

## Usage

### Basic Setup

```python
from rakam_systems_agent.server import run_agent_mcp
from rakam_systems_core.base import BaseComponent

# Create some components
class SearchTool(BaseComponent):
    def run(self, query: str, max_results: int = 10):
        # Your search implementation
        return f"Found {max_results} results for: {query}"

class CalculatorTool(BaseComponent):
    def run(self, operation: str, a: float, b: float):
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        return None

# Initialize components
search = SearchTool(name="search")
calculator = CalculatorTool(name="calculator")

# Create MCP server with components
mcp_server = run_agent_mcp([search, calculator])

# List available components
print(mcp_server.list_components())
# Output: ['calculator', 'search']
```

### Using with Synchronous Messages

```python
# Invoke search tool
result = mcp_server.send_message(
    sender="agent",
    receiver="search",
    message={
        'arguments': {
            'query': 'machine learning',
            'max_results': 5
        }
    }
)
print(result)
# Output: "Found 5 results for: machine learning"

# Invoke calculator tool
result = mcp_server.send_message(
    sender="agent",
    receiver="calculator",
    message={
        'arguments': {
            'operation': 'add',
            'a': 10,
            'b': 5
        }
    }
)
print(result)  # Output: 15
```

### Using with Async Messages

```python
import asyncio

async def main():
    # Async tool invocation
    result = await mcp_server.asend_message(
        sender="agent",
        receiver="search",
        message={
            'arguments': {
                'query': 'deep learning',
                'max_results': 3
            }
        }
    )
    print(result)

asyncio.run(main())
```

### Concurrent Operations

```python
import asyncio

async def concurrent_operations():
    # Execute multiple tool invocations concurrently
    results = await asyncio.gather(
        mcp_server.asend_message(
            sender="agent",
            receiver="search",
            message={'arguments': {'query': 'AI', 'max_results': 5}}
        ),
        mcp_server.asend_message(
            sender="agent",
            receiver="calculator",
            message={'arguments': {'operation': 'multiply', 'a': 7, 'b': 6}}
        ),
        mcp_server.asend_message(
            sender="agent",
            receiver="calculator",
            message={'arguments': {'operation': 'add', 'a': 100, 'b': 200}}
        )
    )
    return results

results = asyncio.run(concurrent_operations())
print(results)
# Output: ["Found 5 results for: AI", 42, 300]
```

### Integration with Tool Registry

```python
from rakam_systems_core.interfaces import ToolRegistry, ToolInvoker

# Create registry and invoker
registry = ToolRegistry()
invoker = ToolInvoker(registry)

# Register MCP server
invoker.register_mcp_server("agent_mcp", mcp_server)

# Register tools
registry.register_mcp_tool(
    name="web_search",
    mcp_server="agent_mcp",
    mcp_tool_name="search",
    description="Search the web for information",
    category="search"
)

registry.register_mcp_tool(
    name="calculate",
    mcp_server="agent_mcp",
    mcp_tool_name="calculator",
    description="Perform mathematical calculations",
    category="math"
)

# Use through invoker
result = await invoker.ainvoke("web_search", query="Python", max_results=10)
result = await invoker.ainvoke("calculate", operation="add", a=5, b=3)
```

### Custom Component with Message Handler

```python
from rakam_systems_core.base import BaseComponent
from typing import Dict, Any

class CustomTool(BaseComponent):
    """Component with custom message handling."""

    def handle_message(self, sender: str, message: Dict[str, Any]):
        """Custom message handler for advanced logic."""
        action = message.get('action', 'default')
        args = message.get('arguments', {})

        if action == 'process':
            return self.process(**args)
        elif action == 'analyze':
            return self.analyze(**args)
        else:
            return self.run(**args)

    def run(self, data: str):
        return f"Default processing: {data}"

    def process(self, data: str, method: str = "standard"):
        return f"Processing {data} with {method} method"

    def analyze(self, data: str):
        return f"Analyzing: {data}"

# Register and use
custom = CustomTool(name="custom_tool")
mcp_server = run_agent_mcp([custom])

# Different actions
result1 = mcp_server.send_message(
    sender="agent",
    receiver="custom_tool",
    message={'action': 'process', 'arguments': {'data': 'test', 'method': 'advanced'}}
)

result2 = mcp_server.send_message(
    sender="agent",
    receiver="custom_tool",
    message={'action': 'analyze', 'arguments': {'data': 'test'}}
)
```

## Function Reference

### run_agent_mcp()

Creates and configures an MCP server with agent components.

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

**Returns:**

- `MCPServer`: Configured MCP server instance with registered components

**Example:**

```python
from rakam_systems_agent.server import run_agent_mcp

tools = [tool1, tool2, tool3]
server = run_agent_mcp(tools, name="my_agent_server", enable_logging=True)
```

## Component Requirements

Components registered with the agent MCP server must:

1. **Inherit from BaseComponent**

   ```python
   from rakam_systems_core.base import BaseComponent

   class MyTool(BaseComponent):
       pass
   ```

2. **Have a unique name**

   ```python
   tool = MyTool(name="unique_tool_name")
   ```

3. **Implement either:**
   - `run(*args, **kwargs)` method for standard invocation
   - `handle_message(sender, message)` method for custom message handling

4. **Support async (optional)**
   ```python
   class AsyncTool(BaseComponent):
       async def run(self, data: str):
           # Async operation
           await asyncio.sleep(1)
           return f"Processed: {data}"
   ```

## Architecture

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
     ┌─────────┼─────────┬─────────┐
     ▼         ▼         ▼         ▼
┌─────────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ Search  │ │Calc  │ │ DB   │ │ ...  │
│ Tool    │ │Tool  │ │Tool  │ │      │
└─────────┘ └──────┘ └──────┘ └──────┘
  (BaseComponent implementations)
```

## Benefits

1. **Decoupled Architecture**: Components don't need direct dependencies
2. **Standardized Interface**: All tools work the same way through MCP
3. **Dynamic Registration**: Add/remove tools at runtime
4. **Tool Discovery**: Query available tools dynamically
5. **Async Support**: Efficient concurrent operations
6. **Flexible Integration**: Works with various agent architectures
7. **Easy Testing**: Mock individual components easily

## Advanced Usage

### Adding Components at Runtime

```python
# Create server with initial components
server = run_agent_mcp([tool1, tool2])

# Add more components later
new_tool = NewTool(name="new_tool")
server.register_component(new_tool)

# Verify registration
assert server.has_component("new_tool")
```

### Removing Components

```python
# Remove a component
server.unregister_component("old_tool")

# Check if removed
assert not server.has_component("old_tool")
```

### Server Statistics

```python
# Get server stats
stats = server.get_stats()
print(f"Server: {stats['name']}")
print(f"Components: {stats['component_count']}")
print(f"Available: {stats['components']}")
```

## Examples

See comprehensive examples in:

- `/app/rakam_systems/rakam_systems/examples/ai_agents_examples/` - Agent examples
- `/docs/MCP_QUICKSTART.md` - Quick start guide
- `/docs/MCP_ARCHITECTURE_DIAGRAM.md` - Architecture overview

## Related Documentation

- [MCP Server API](/app/rakam_systems/rakam_systems/ai_core/mcp/README.md)
- [MCP Quickstart](/docs/MCP_QUICKSTART.md)
- [Vector Store MCP Server](/app/rakam_systems/rakam_systems/ai_vectorstore/server/README.md)
- [BaseComponent API](/app/rakam_systems/rakam_systems/ai_core/base.py)

## Comparison with Vector Store MCP Server

| Feature            | Agent MCP                   | Vector Store MCP                  |
| ------------------ | --------------------------- | --------------------------------- |
| **Purpose**        | General agent components    | Vector store operations           |
| **Components**     | User-provided               | Pre-built (search, storage, info) |
| **Registration**   | Dynamic, flexible           | Automatic with vector store       |
| **Use Case**       | Custom tools, utilities     | Document search, RAG systems      |
| **Initialization** | `run_agent_mcp(components)` | `run_vector_mcp(vector_store)`    |

Both servers use the same underlying `MCPServer` and can be used together in a single system.

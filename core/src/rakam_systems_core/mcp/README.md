# MCP Server Module

## Overview

The `mcp_server` module provides a Model Context Protocol (MCP) server implementation for message-based component communication. It acts as a lightweight message router that enables loose coupling between components.

## Features

✅ **Component Registration** - Register and manage multiple components  
✅ **Message Routing** - Route messages between components using sender/receiver pattern  
✅ **Async Support** - Full support for both synchronous and asynchronous operations  
✅ **Flexible Handlers** - Components can implement custom message handlers  
✅ **Auto Argument Extraction** - Automatically extracts and passes arguments from messages  
✅ **Error Handling** - Comprehensive error handling with logging  
✅ **Component Discovery** - Query and list registered components

## Quick Start

### Basic Usage

```python
from rakam_systems_core.mcp.mcp_server import MCPServer
from rakam_systems_core.base import BaseComponent

# Create server
server = MCPServer(name="my_server")
server.setup()

# Create a component
class MyTool(BaseComponent):
    def run(self, value: int):
        return {'result': value * 2}

# Register component
tool = MyTool(name="calculator")
server.register_component(tool)

# Send message
result = server.send_message(
    sender="client",
    receiver="calculator",
    message={'arguments': {'value': 5}}
)
# result = {'result': 10}
```

### Async Usage

```python
import asyncio

# Async component
class AsyncTool(BaseComponent):
    async def run(self, query: str):
        await asyncio.sleep(0.1)  # Simulate async work
        return {'query': query, 'results': [...]}

# Register
tool = AsyncTool(name="search")
server.register_component(tool)

# Send async message
result = await server.asend_message(
    sender="client",
    receiver="search",
    message={'arguments': {'query': 'test'}}
)
```

## API Reference

### MCPServer

#### Constructor

```python
MCPServer(
    name: str = "mcp_server",
    config: Optional[Dict[str, Any]] = None,
    enable_logging: bool = True
)
```

**Parameters:**

- `name` (str): Name of the MCP server
- `config` (Dict, optional): Configuration dictionary
- `enable_logging` (bool): Whether to enable detailed logging

**Example:**

```python
server = MCPServer(
    name="production_server",
    enable_logging=True
)
```

#### Methods

##### `register_component(component: BaseComponent) -> None`

Register a component with the server.

```python
server.register_component(my_component)
```

##### `unregister_component(component_name: str) -> bool`

Unregister a component.

```python
success = server.unregister_component("my_component")
```

##### `send_message(sender: str, receiver: str, message: Dict[str, Any]) -> Any`

Send a synchronous message to a component.

```python
result = server.send_message(
    sender="client",
    receiver="tool_name",
    message={
        'action': 'invoke_tool',
        'arguments': {'param': 'value'}
    }
)
```

##### `async asend_message(sender: str, receiver: str, message: Dict[str, Any]) -> Any`

Send an asynchronous message to a component.

```python
result = await server.asend_message(
    sender="client",
    receiver="tool_name",
    message={
        'action': 'invoke_tool',
        'arguments': {'param': 'value'}
    }
)
```

##### `get_component(component_name: str) -> Optional[BaseComponent]`

Get a registered component by name.

```python
component = server.get_component("tool_name")
```

##### `list_components() -> List[str]`

List all registered component names (sorted).

```python
names = server.list_components()
# ['component1', 'component2', ...]
```

##### `has_component(component_name: str) -> bool`

Check if a component is registered.

```python
if server.has_component("tool_name"):
    # Component is registered
    pass
```

##### `get_stats() -> Dict[str, Any]`

Get server statistics.

```python
stats = server.get_stats()
# {
#     'name': 'my_server',
#     'component_count': 5,
#     'components': ['comp1', 'comp2', ...],
#     'logging_enabled': True
# }
```

## Message Format

Messages should follow this structure:

```python
message = {
    'action': 'invoke_tool',  # Optional, describes the action
    'tool_name': 'my_tool',   # Optional, tool identifier
    'arguments': {            # Required for components without custom handlers
        'param1': 'value1',
        'param2': 'value2'
    }
}
```

The server automatically extracts `arguments` and passes them to the component's `run()` method.

## Component Implementation

### Basic Component

```python
from rakam_systems_core.base import BaseComponent

class MyComponent(BaseComponent):
    """Component with automatic argument extraction."""

    def run(self, param1: str, param2: int = 0):
        # Arguments are automatically extracted from message
        return {
            'param1': param1,
            'param2': param2,
            'processed': True
        }
```

### Component with Custom Handler

```python
class CustomHandlerComponent(BaseComponent):
    """Component with custom message handling."""

    def handle_message(self, sender: str, message: Dict[str, Any]):
        # Custom logic for handling messages
        action = message.get('action')

        if action == 'special_action':
            return self._handle_special(message)
        else:
            # Fall back to default behavior
            return self.run(**message.get('arguments', {}))

    def _handle_special(self, message):
        return {'status': 'special_handled'}

    def run(self, **kwargs):
        return {'status': 'normal'}
```

### Async Component

```python
class AsyncComponent(BaseComponent):
    """Async component."""

    async def run(self, query: str):
        # Perform async operations
        results = await self._async_operation(query)
        return {'results': results}

    async def _async_operation(self, query):
        await asyncio.sleep(0.1)
        return [query, query.upper()]
```

### Async Custom Handler

```python
class AsyncHandlerComponent(BaseComponent):
    """Async component with custom handler."""

    async def handle_message(self, sender: str, message: Dict[str, Any]):
        # Async custom handling
        await asyncio.sleep(0.01)
        return {
            'sender': sender,
            'handled': True,
            'async': True
        }

    async def run(self, **kwargs):
        return {'handled': False}
```

## Usage Patterns

### Pattern 1: Simple Tool Registry

```python
# Server as a tool registry
server = MCPServer(name="tool_registry")
server.setup()

# Register tools
server.register_component(SearchTool(name="search"))
server.register_component(CalculatorTool(name="calc"))
server.register_component(DatabaseTool(name="db"))

# List available tools
available = server.list_components()
print(f"Available tools: {available}")
```

### Pattern 2: Request-Response Pattern

```python
# Client sends request
response = server.send_message(
    sender="user_client",
    receiver="search",
    message={
        'action': 'search',
        'arguments': {
            'query': 'machine learning',
            'top_k': 5
        }
    }
)

# Process response
for result in response['results']:
    print(result)
```

### Pattern 3: Async Concurrent Operations

```python
# Execute multiple operations concurrently
results = await asyncio.gather(
    server.asend_message(
        sender="client",
        receiver="search",
        message={'arguments': {'query': 'AI'}}
    ),
    server.asend_message(
        sender="client",
        receiver="database",
        message={'arguments': {'table': 'users'}}
    ),
    server.asend_message(
        sender="client",
        receiver="calculator",
        message={'arguments': {'value': 42}}
    )
)
```

### Pattern 4: Tool Invoker Integration

```python
from rakam_systems_core.interfaces import ToolRegistry, ToolInvoker

# Create MCP server
server = MCPServer(name="tool_server")
server.setup()

# Register components
server.register_component(SearchTool(name="search"))

# Register in tool registry
registry = ToolRegistry()
registry.register_mcp_tool(
    name="mcp_search",
    mcp_server="tool_server",
    mcp_tool_name="search",
    description="Search documents"
)

# Create invoker
invoker = ToolInvoker(registry)
invoker.register_mcp_server("tool_server", server)

# Invoke via tool system
result = await invoker.ainvoke(
    "mcp_search",
    query="test",
    top_k=5
)
```

## Error Handling

The server provides comprehensive error handling:

### Component Not Found

```python
try:
    result = server.send_message(
        sender="client",
        receiver="nonexistent",
        message={'arguments': {}}
    )
except KeyError as e:
    print(f"Component not found: {e}")
    # Lists available components in error message
```

### Component Errors

```python
# Errors in components are propagated
class ErrorComponent(BaseComponent):
    def run(self):
        raise ValueError("Something went wrong")

server.register_component(ErrorComponent(name="error"))

try:
    server.send_message(
        sender="client",
        receiver="error",
        message={'arguments': {}}
    )
except ValueError as e:
    print(f"Component error: {e}")
```

## Best Practices

### 1. Descriptive Names

```python
# Good
server = MCPServer(name="production_vector_store_server")

# Avoid
server = MCPServer(name="server1")
```

### 2. Consistent Message Format

```python
# Consistent structure
message = {
    'action': 'invoke_tool',
    'arguments': {
        'param1': 'value1',
        'param2': 'value2'
    }
}
```

### 3. Error Handling

```python
async def safe_message_send(server, receiver, message):
    """Wrapper with error handling."""
    try:
        return await server.asend_message(
            sender="client",
            receiver=receiver,
            message=message
        )
    except KeyError:
        print(f"Component '{receiver}' not found")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### 4. Component Validation

```python
# Check before sending
if server.has_component("search"):
    result = server.send_message(...)
else:
    print("Search component not available")
```

### 5. Resource Management

```python
# Unregister when done
try:
    # Use component
    result = server.send_message(...)
finally:
    # Clean up
    server.unregister_component("temp_component")
```

## Testing

The module includes comprehensive tests. Run them with:

```bash
pytest app/rakam_systems/tests/test_mcp_server.py -v
```

Tests cover:

- Component registration/unregistration
- Synchronous message routing
- Asynchronous message routing
- Error handling
- Component discovery
- Custom handlers
- Mixed async/sync operations

## Advanced Features

### Logging Control

```python
# Enable detailed logging
server = MCPServer(name="debug_server", enable_logging=True)

# Disable logging for production
server = MCPServer(name="prod_server", enable_logging=False)
```

### Statistics Monitoring

```python
# Get server stats
stats = server.get_stats()

print(f"Server: {stats['name']}")
print(f"Components: {stats['component_count']}")
print(f"Component list: {stats['components']}")
```

### Dynamic Component Management

```python
# Hot-swap components
server.unregister_component("old_version")
server.register_component(NewVersion(name="service"))

# Conditional registration
if os.getenv("ENABLE_FEATURE_X"):
    server.register_component(FeatureX(name="feature_x"))
```

## See Also

- **Tool System**: `app/rakam_systems/rakam_systems/ai_core/interfaces/tool_registry.py`
- **Tool Invoker**: `app/rakam_systems/rakam_systems/ai_core/interfaces/tool_invoker.py`
- **Base Component**: `app/rakam_systems/rakam_systems/ai_core/base.py`
- **Examples**: `app/rakam_systems/rakam_systems/examples/ai_agents_examples/tool_system_example.py`
- **MCP Guide**: `docs/MCP_VECTOR_STORE_GUIDE.md`

## License

Part of the rakam_systems framework.

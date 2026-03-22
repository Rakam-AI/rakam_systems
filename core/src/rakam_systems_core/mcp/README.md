# MCP Server Module

## Overview

The `mcp_server` module provides a Model Context Protocol (MCP) server implementation for message-based component communication. It acts as a lightweight message router that enables loose coupling between components.

## Features

- **Component Registration** — Register and manage multiple components
- **Message Routing** — Route messages between components using sender/receiver pattern
- **Async Support** — Full support for both synchronous and asynchronous operations
- **Flexible Handlers** — Components can implement custom message handlers
- **Auto Argument Extraction** — Automatically extracts and passes arguments from messages
- **Component Discovery** — Query and list registered components

## Quick Start

```python
from rakam_systems_core.mcp.mcp_server import MCPServer
from rakam_systems_core.base import BaseComponent

server = MCPServer(name="my_server")
server.setup()

class MyTool(BaseComponent):
    def run(self, value: int):
        return {'result': value * 2}

tool = MyTool(name="calculator")
server.register_component(tool)

result = server.send_message(
    sender="client",
    receiver="calculator",
    message={'arguments': {'value': 5}}
)
# result = {'result': 10}
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

#### Methods

| Method | Description |
|--------|-------------|
| `register_component(component)` | Register a `BaseComponent` with the server |
| `unregister_component(name)` | Unregister a component by name |
| `send_message(sender, receiver, message)` | Send a synchronous message |
| `asend_message(sender, receiver, message)` | Send an asynchronous message |
| `get_component(name)` | Get a registered component by name |
| `list_components()` | List all registered component names (sorted) |
| `has_component(name)` | Check if a component is registered |
| `get_stats()` | Get server statistics |

## Message Format

```python
message = {
    'action': 'invoke_tool',       # Optional
    'arguments': {                  # Required for components without custom handlers
        'param1': 'value1',
        'param2': 'value2'
    }
}
```

The server automatically extracts `arguments` and passes them to the component's `run()` method.

## Component Implementation

### Basic Component

```python
class MyComponent(BaseComponent):
    def run(self, param1: str, param2: int = 0):
        return {'param1': param1, 'param2': param2, 'processed': True}
```

### Component with Custom Handler

```python
class CustomComponent(BaseComponent):
    def handle_message(self, sender: str, message: Dict[str, Any]):
        action = message.get('action')
        if action == 'special_action':
            return self._handle_special(message)
        return self.run(**message.get('arguments', {}))

    def run(self, **kwargs):
        return {'status': 'normal'}
```

### Async Component

```python
class AsyncComponent(BaseComponent):
    async def run(self, query: str):
        results = await self._async_operation(query)
        return {'results': results}
```

## Testing

```bash
pytest tests/ -v
```

## See Also

- **Tool System**: [`tool_registry.py`](../interfaces/tool_registry.py)
- **Tool Invoker**: [`tool_invoker.py`](../interfaces/tool_invoker.py)
- **Base Component**: [`base.py`](../base.py)
- [Official documentation](https://rakam-ai.github.io/rakam-systems-docs/)

## License

Part of the rakam_systems framework.

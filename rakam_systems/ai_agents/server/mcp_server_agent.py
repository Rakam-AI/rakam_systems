from __future__ import annotations
from typing import Iterable
from ai_core.mcp.mcp_server import MCPServer
from ai_core.base import BaseComponent

def run_agent_mcp(components: Iterable[BaseComponent]) -> MCPServer:
    """Create a minimal MCP-like registry and register given components.
    Returns the in-process server instance so callers can send messages.
    """
    server = MCPServer(name="agent_mcp")
    server.setup()
    for c in components:
        server.register_component(c)
    return server

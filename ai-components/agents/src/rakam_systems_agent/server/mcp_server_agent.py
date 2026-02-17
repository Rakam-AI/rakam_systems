"""
Agent MCP Server

This module provides an MCP (Model Context Protocol) server for AI agent components.
It creates a unified interface for registering and managing agent-related tools and
components that can communicate through a standardized message-based protocol.

Features:
- Dynamic component registration
- Message-based communication between components
- Support for any BaseComponent implementation
- Async/await support for concurrent operations
- Easy integration with AI agents and tool systems

Example:
    >>> from rakam_systems_agent.server.mcp_server_agent import run_agent_mcp
    >>> from rakam_systems_core.base import BaseComponent
    >>> 
    >>> # Create some agent components
    >>> class SearchTool(BaseComponent):
    ...     def run(self, query: str):
    ...         return f"Searching for: {query}"
    >>> 
    >>> search_tool = SearchTool(name="search")
    >>> 
    >>> # Create MCP server with components
    >>> mcp_server = run_agent_mcp([search_tool])
    >>> 
    >>> # Use the server to route messages
    >>> result = mcp_server.send_message(
    ...     sender="agent",
    ...     receiver="search",
    ...     message={'arguments': {'query': 'test'}}
    ... )
"""

from __future__ import annotations
from typing import Iterable, Optional

from rakam_systems_tools.utils import logging

from rakam_systems_core.mcp.mcp_server import MCPServer
from rakam_systems_core.base import BaseComponent

logger = logging.getLogger(__name__)


def run_agent_mcp(
    components: Iterable[BaseComponent],
    name: str = "agent_mcp",
    enable_logging: bool = False
) -> MCPServer:
    """
    Create and configure an MCP server with agent components.

    This function creates a fully configured MCP server and registers all provided
    components, making them available for message-based communication. Components
    can be tools, utilities, or any BaseComponent implementation.

    Args:
        components: Iterable of BaseComponent instances to register
        name: Name for the MCP server (default: "agent_mcp")
        enable_logging: Whether to enable detailed MCP logging (default: False)

    Returns:
        MCPServer instance with registered components

    Example:
        >>> from rakam_systems_agent.server import run_agent_mcp
        >>> from rakam_systems_core.base import BaseComponent
        >>> 
        >>> # Create components
        >>> class CalculatorTool(BaseComponent):
        ...     def run(self, operation: str, a: float, b: float):
        ...         if operation == "add":
        ...             return a + b
        ...         elif operation == "multiply":
        ...             return a * b
        >>> 
        >>> calculator = CalculatorTool(name="calculator")
        >>> 
        >>> # Create MCP server
        >>> mcp_server = run_agent_mcp([calculator])
        >>> 
        >>> # List registered components
        >>> print(mcp_server.list_components())
        >>> # Output: ['calculator']
        >>> 
        >>> # Use the server
        >>> result = mcp_server.send_message(
        ...     sender="agent",
        ...     receiver="calculator",
        ...     message={'arguments': {'operation': 'add', 'a': 5, 'b': 3}}
        ... )
        >>> print(result)  # Output: 8

    Example with async:
        >>> result = await mcp_server.asend_message(
        ...     sender="agent",
        ...     receiver="calculator",
        ...     message={'arguments': {'operation': 'multiply', 'a': 5, 'b': 3}}
        ... )
        >>> print(result)  # Output: 15

    Note:
        - All components must be BaseComponent instances with a unique name
        - Components should implement the run() method or handle_message() method
        - The server supports both sync (send_message) and async (asend_message) operations
    """
    logger.info(f"Creating agent MCP server: {name}")

    # Create MCP server
    server = MCPServer(name=name, enable_logging=enable_logging)
    server.setup()

    # Register all components
    component_count = 0
    for component in components:
        server.register_component(component)
        component_count += 1

    logger.info(
        f"Agent MCP server '{name}' ready with {component_count} component(s): "
        f"{server.list_components()}"
    )

    return server

"""
Tool Invoker for uniform tool invocation across different modes.
Supports both direct tool calls and MCP-based tool calls.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Union
import asyncio
from .tool_registry import ToolRegistry, ToolMetadata, ToolMode
from .tool import Tool, ToolComponent


class ToolInvocationError(Exception):
    """Base exception for tool invocation errors."""
    pass


class ToolNotFoundError(ToolInvocationError):
    """Raised when a tool is not found in the registry."""
    pass


class MCPServerNotFoundError(ToolInvocationError):
    """Raised when an MCP server is not available."""
    pass


class ToolInvoker:
    """
    Uniform interface for invoking tools regardless of their execution mode.
    
    Supports:
    - Direct tool invocation (synchronous and asynchronous)
    - MCP-based tool invocation via registered servers
    - Automatic mode selection based on tool registration
    - Error handling and validation
    
    Example:
        >>> registry = ToolRegistry()
        >>> invoker = ToolInvoker(registry)
        >>> 
        >>> # Invoke a direct tool
        >>> result = await invoker.ainvoke("calculate", x=10, y=20)
        >>> 
        >>> # Invoke an MCP tool (with MCP server registered)
        >>> invoker.register_mcp_server("search_server", mcp_server_instance)
        >>> result = await invoker.ainvoke("web_search", query="Python")
    """
    
    def __init__(self, registry: ToolRegistry):
        """
        Initialize the ToolInvoker.
        
        Args:
            registry: ToolRegistry instance containing registered tools
        """
        self.registry = registry
        self._mcp_servers: Dict[str, Any] = {}
    
    def register_mcp_server(self, server_name: str, server_instance: Any) -> None:
        """
        Register an MCP server for tool invocation.
        
        Args:
            server_name: Name of the MCP server
            server_instance: The MCP server instance
        """
        self._mcp_servers[server_name] = server_instance
    
    def unregister_mcp_server(self, server_name: str) -> bool:
        """
        Unregister an MCP server.
        
        Returns:
            True if server was unregistered, False if not found
        """
        if server_name in self._mcp_servers:
            del self._mcp_servers[server_name]
            return True
        return False
    
    def invoke(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Synchronously invoke a tool by name.
        
        Args:
            tool_name: Name of the tool to invoke
            **kwargs: Arguments to pass to the tool
        
        Returns:
            Result from the tool execution
        
        Raises:
            ToolNotFoundError: If tool is not registered
            ToolInvocationError: If tool invocation fails
        """
        metadata = self.registry.get_tool(tool_name)
        if metadata is None:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found in registry")
        
        if metadata.mode == ToolMode.DIRECT:
            return self._invoke_direct(metadata, **kwargs)
        elif metadata.mode == ToolMode.MCP:
            return self._invoke_mcp(metadata, **kwargs)
        else:
            raise ToolInvocationError(f"Unknown tool mode: {metadata.mode}")
    
    async def ainvoke(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Asynchronously invoke a tool by name.
        
        Args:
            tool_name: Name of the tool to invoke
            **kwargs: Arguments to pass to the tool
        
        Returns:
            Result from the tool execution
        
        Raises:
            ToolNotFoundError: If tool is not registered
            ToolInvocationError: If tool invocation fails
        """
        metadata = self.registry.get_tool(tool_name)
        if metadata is None:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found in registry")
        
        if metadata.mode == ToolMode.DIRECT:
            return await self._ainvoke_direct(metadata, **kwargs)
        elif metadata.mode == ToolMode.MCP:
            return await self._ainvoke_mcp(metadata, **kwargs)
        else:
            raise ToolInvocationError(f"Unknown tool mode: {metadata.mode}")
    
    def _invoke_direct(self, metadata: ToolMetadata, **kwargs: Any) -> Any:
        """Invoke a direct tool synchronously."""
        tool_instance = metadata.tool_instance
        
        if isinstance(tool_instance, Tool):
            # Tool wrapper - call the function directly
            return tool_instance(**kwargs)
        elif isinstance(tool_instance, ToolComponent):
            # ToolComponent - use run method
            # Convert kwargs to appropriate format for ToolComponent
            query = kwargs.get('query', '')
            return tool_instance.run(query)
        else:
            raise ToolInvocationError(
                f"Invalid tool instance type for '{metadata.name}': {type(tool_instance)}"
            )
    
    async def _ainvoke_direct(self, metadata: ToolMetadata, **kwargs: Any) -> Any:
        """Invoke a direct tool asynchronously."""
        tool_instance = metadata.tool_instance
        
        if isinstance(tool_instance, Tool):
            # Tool wrapper - use acall for async support
            return await tool_instance.acall(**kwargs)
        elif isinstance(tool_instance, ToolComponent):
            # ToolComponent - check if it has async run
            query = kwargs.get('query', '')
            if hasattr(tool_instance, 'arun'):
                return await tool_instance.arun(query)
            else:
                # Fall back to sync run in thread pool
                return await asyncio.get_event_loop().run_in_executor(
                    None, tool_instance.run, query
                )
        else:
            raise ToolInvocationError(
                f"Invalid tool instance type for '{metadata.name}': {type(tool_instance)}"
            )
    
    def _invoke_mcp(self, metadata: ToolMetadata, **kwargs: Any) -> Any:
        """Invoke a tool via MCP server synchronously."""
        if metadata.mcp_server not in self._mcp_servers:
            raise MCPServerNotFoundError(
                f"MCP server '{metadata.mcp_server}' not registered"
            )
        
        server = self._mcp_servers[metadata.mcp_server]
        tool_name = metadata.mcp_tool_name or metadata.name
        
        # Construct MCP message
        message = {
            'action': 'invoke_tool',
            'tool_name': tool_name,
            'arguments': kwargs,
        }
        
        try:
            # Send message to the tool component (receiver is the tool name)
            result = server.send_message(
                sender='tool_invoker',
                receiver=tool_name,
                message=message
            )
            return result
        except Exception as e:
            raise ToolInvocationError(
                f"Failed to invoke MCP tool '{metadata.name}': {str(e)}"
            ) from e
    
    async def _ainvoke_mcp(self, metadata: ToolMetadata, **kwargs: Any) -> Any:
        """Invoke a tool via MCP server asynchronously."""
        if metadata.mcp_server not in self._mcp_servers:
            raise MCPServerNotFoundError(
                f"MCP server '{metadata.mcp_server}' not registered"
            )
        
        server = self._mcp_servers[metadata.mcp_server]
        tool_name = metadata.mcp_tool_name or metadata.name
        
        # Construct MCP message
        message = {
            'action': 'invoke_tool',
            'tool_name': tool_name,
            'arguments': kwargs,
        }
        
        try:
            # Check if server has async send_message
            if hasattr(server, 'asend_message'):
                result = await server.asend_message(
                    sender='tool_invoker',
                    receiver=tool_name,
                    message=message
                )
            else:
                # Fall back to sync in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    server.send_message,
                    'tool_invoker',
                    tool_name,
                    message
                )
            return result
        except Exception as e:
            raise ToolInvocationError(
                f"Failed to invoke MCP tool '{metadata.name}': {str(e)}"
            ) from e
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            Dictionary with tool information or None if not found
        """
        metadata = self.registry.get_tool(tool_name)
        if metadata is None:
            return None
        
        return metadata.to_dict()
    
    def list_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available tools with their information.
        
        Returns:
            Dictionary mapping tool names to their metadata
        """
        return {
            metadata.name: metadata.to_dict()
            for metadata in self.registry.get_all_tools()
        }
    
    def __repr__(self) -> str:
        return (
            f"ToolInvoker(tools={len(self.registry)}, "
            f"mcp_servers={len(self._mcp_servers)})"
        )


"""
Tool Registry for managing and discovering tools across the system.
Provides a centralized registry for both direct tool invocation and MCP-based tools.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable, Union
from .tool import ToolComponent


class ToolMode:
    """Constants for tool invocation modes."""
    DIRECT = "direct"
    MCP = "mcp"


class ToolMetadata:
    """Metadata for a registered tool."""
    
    def __init__(
        self,
        name: str,
        description: str,
        mode: str,
        tool_instance: Optional[ToolComponent] = None,
        mcp_server: Optional[str] = None,
        mcp_tool_name: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.mode = mode
        self.tool_instance = tool_instance
        self.mcp_server = mcp_server
        self.mcp_tool_name = mcp_tool_name
        self.category = category or "general"
        self.tags = tags or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "mcp_server": self.mcp_server,
            "mcp_tool_name": self.mcp_tool_name,
            "category": self.category,
            "tags": self.tags,
        }


class ToolRegistry:
    """
    Central registry for managing tools in the system.
    
    Supports:
    - Direct tool registration (callable functions or Tool instances)
    - MCP-based tool registration (tools exposed via MCP servers)
    - Tool discovery and lookup by name, category, or tags
    - Automatic tool conversion for different agent frameworks
    
    Example:
        >>> registry = ToolRegistry()
        >>> 
        >>> # Register a direct tool
        >>> def calculate(x: int, y: int) -> int:
        ...     return x + y
        >>> registry.register_direct_tool(
        ...     name="calculate",
        ...     function=calculate,
        ...     description="Add two numbers",
        ...     json_schema={...}
        ... )
        >>>
        >>> # Register an MCP tool
        >>> registry.register_mcp_tool(
        ...     name="search",
        ...     mcp_server="search_server",
        ...     mcp_tool_name="web_search",
        ...     description="Search the web"
        ... )
        >>>
        >>> # Get all tools
        >>> tools = registry.get_all_tools()
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolMetadata] = {}
        self._categories: Dict[str, List[str]] = {}
        self._tags: Dict[str, List[str]] = {}
    
    def register_direct_tool(
        self,
        name: str,
        function: Callable[..., Any],
        description: str,
        json_schema: Dict[str, Any],
        takes_ctx: bool = False,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a tool for direct invocation.
        
        Args:
            name: Unique name for the tool
            function: The callable function
            description: Human-readable description
            json_schema: JSON schema for the tool parameters
            takes_ctx: Whether the tool takes context as first argument
            category: Optional category for organizing tools
            tags: Optional tags for filtering tools
        """
        # Create a ToolComponent from the function
        tool = ToolComponent.from_function(
            function=function,
            name=name,
            description=description,
            json_schema=json_schema,
            takes_ctx=takes_ctx,
        )
        
        metadata = ToolMetadata(
            name=name,
            description=description,
            mode=ToolMode.DIRECT,
            tool_instance=tool,
            category=category,
            tags=tags,
        )
        
        self._register_metadata(metadata)
    
    def register_tool_instance(
        self,
        tool: ToolComponent,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register an existing ToolComponent instance.
        
        Args:
            tool: ToolComponent instance
            category: Optional category for organizing tools
            tags: Optional tags for filtering tools
        """
        if not isinstance(tool, ToolComponent):
            raise ValueError(f"Unsupported tool type: {type(tool)}. Expected ToolComponent.")
        
        metadata = ToolMetadata(
            name=tool.name,
            description=tool.description,
            mode=ToolMode.DIRECT,
            tool_instance=tool,
            category=category,
            tags=tags,
        )
        
        self._register_metadata(metadata)
    
    def register_mcp_tool(
        self,
        name: str,
        mcp_server: str,
        mcp_tool_name: str,
        description: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a tool that will be invoked via MCP server.
        
        Args:
            name: Unique name for the tool in this registry
            mcp_server: Name of the MCP server hosting the tool
            mcp_tool_name: Name of the tool on the MCP server
            description: Human-readable description
            category: Optional category for organizing tools
            tags: Optional tags for filtering tools
        """
        metadata = ToolMetadata(
            name=name,
            description=description,
            mode=ToolMode.MCP,
            mcp_server=mcp_server,
            mcp_tool_name=mcp_tool_name,
            category=category,
            tags=tags,
        )
        
        self._register_metadata(metadata)
    
    def _register_metadata(self, metadata: ToolMetadata) -> None:
        """Internal method to register tool metadata and update indices."""
        if metadata.name in self._tools:
            raise ValueError(f"Tool '{metadata.name}' is already registered")
        
        self._tools[metadata.name] = metadata
        
        # Update category index
        if metadata.category not in self._categories:
            self._categories[metadata.category] = []
        self._categories[metadata.category].append(metadata.name)
        
        # Update tag index
        for tag in metadata.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            self._tags[tag].append(metadata.name)
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name."""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[ToolMetadata]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_tools_by_category(self, category: str) -> List[ToolMetadata]:
        """Get all tools in a specific category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names]
    
    def get_tools_by_tag(self, tag: str) -> List[ToolMetadata]:
        """Get all tools with a specific tag."""
        tool_names = self._tags.get(tag, [])
        return [self._tools[name] for name in tool_names]
    
    def get_tools_by_mode(self, mode: str) -> List[ToolMetadata]:
        """Get all tools for a specific invocation mode."""
        return [tool for tool in self._tools.values() if tool.mode == mode]
    
    def list_categories(self) -> List[str]:
        """List all registered categories."""
        return list(self._categories.keys())
    
    def list_tags(self) -> List[str]:
        """List all registered tags."""
        return list(self._tags.keys())
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool by name.
        
        Returns:
            True if tool was unregistered, False if not found
        """
        if name not in self._tools:
            return False
        
        metadata = self._tools[name]
        
        # Remove from indices
        if metadata.category in self._categories:
            self._categories[metadata.category].remove(name)
            if not self._categories[metadata.category]:
                del self._categories[metadata.category]
        
        for tag in metadata.tags:
            if tag in self._tags:
                self._tags[tag].remove(name)
                if not self._tags[tag]:
                    del self._tags[tag]
        
        del self._tools[name]
        return True
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._categories.clear()
        self._tags.clear()
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self._tools)}, categories={len(self._categories)}, tags={len(self._tags)})"


"""
Tool Loader for automatic tool discovery and registration from configuration.
Supports loading tools from YAML/JSON config files and Python modules.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
import importlib
import inspect
import yaml
import json
from pathlib import Path
from .tool_registry import ToolRegistry
from .tool import Tool


class ToolLoadError(Exception):
    """Base exception for tool loading errors."""
    pass


class ToolLoader:
    """
    Load and register tools from configuration files.
    
    Supports:
    - YAML and JSON configuration files
    - Auto-discovery of tool functions from Python modules
    - Dynamic import and registration
    - Validation of tool definitions
    
    Configuration Format (YAML):
        tools:
          - name: calculate
            type: direct
            module: my_tools.math_tools
            function: calculate
            description: Add two numbers
            category: math
            tags: [arithmetic, basic]
            schema:
              type: object
              properties:
                x:
                  type: integer
                  description: First number
                y:
                  type: integer
                  description: Second number
              required: [x, y]
          
          - name: web_search
            type: mcp
            mcp_server: search_server
            mcp_tool_name: search
            description: Search the web
            category: search
            tags: [web, external]
    
    Example:
        >>> loader = ToolLoader(registry)
        >>> loader.load_from_yaml("tools.yaml")
        >>> print(f"Loaded {len(registry)} tools")
    """
    
    def __init__(self, registry: ToolRegistry):
        """
        Initialize the ToolLoader.
        
        Args:
            registry: ToolRegistry instance to register tools into
        """
        self.registry = registry
    
    def load_from_yaml(self, file_path: str) -> int:
        """
        Load tools from a YAML configuration file.
        
        Args:
            file_path: Path to YAML configuration file
        
        Returns:
            Number of tools loaded
        
        Raises:
            ToolLoadError: If file cannot be loaded or parsed
        """
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            return self.load_from_config(config)
        except FileNotFoundError:
            raise ToolLoadError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ToolLoadError(f"Failed to parse YAML file: {str(e)}") from e
    
    def load_from_json(self, file_path: str) -> int:
        """
        Load tools from a JSON configuration file.
        
        Args:
            file_path: Path to JSON configuration file
        
        Returns:
            Number of tools loaded
        
        Raises:
            ToolLoadError: If file cannot be loaded or parsed
        """
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            return self.load_from_config(config)
        except FileNotFoundError:
            raise ToolLoadError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ToolLoadError(f"Failed to parse JSON file: {str(e)}") from e
    
    def load_from_config(self, config: Dict[str, Any]) -> int:
        """
        Load tools from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Number of tools loaded
        
        Raises:
            ToolLoadError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ToolLoadError("Configuration must be a dictionary")
        
        tools_config = config.get('tools', [])
        if not isinstance(tools_config, list):
            raise ToolLoadError("'tools' must be a list")
        
        loaded_count = 0
        errors = []
        
        for tool_config in tools_config:
            try:
                self._load_tool(tool_config)
                loaded_count += 1
            except Exception as e:
                tool_name = tool_config.get('name', 'unknown')
                errors.append(f"Failed to load tool '{tool_name}': {str(e)}")
        
        if errors:
            error_msg = "\n".join(errors)
            raise ToolLoadError(f"Errors loading tools:\n{error_msg}")
        
        return loaded_count
    
    def _load_tool(self, tool_config: Dict[str, Any]) -> None:
        """Load a single tool from configuration."""
        # Validate required fields
        if 'name' not in tool_config:
            raise ToolLoadError("Tool configuration must have 'name' field")
        
        name = tool_config['name']
        tool_type = tool_config.get('type', 'direct')
        description = tool_config.get('description', f"Tool: {name}")
        category = tool_config.get('category')
        tags = tool_config.get('tags', [])
        
        if tool_type == 'direct':
            self._load_direct_tool(name, tool_config, description, category, tags)
        elif tool_type == 'mcp':
            self._load_mcp_tool(name, tool_config, description, category, tags)
        else:
            raise ToolLoadError(f"Unknown tool type: {tool_type}")
    
    def _load_direct_tool(
        self,
        name: str,
        config: Dict[str, Any],
        description: str,
        category: Optional[str],
        tags: List[str]
    ) -> None:
        """Load a direct tool from configuration."""
        # Get module and function name
        module_name = config.get('module')
        function_name = config.get('function')
        
        if not module_name or not function_name:
            raise ToolLoadError(
                f"Direct tool '{name}' must specify 'module' and 'function'"
            )
        
        # Import the function
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
        except ImportError as e:
            raise ToolLoadError(
                f"Failed to import module '{module_name}': {str(e)}"
            ) from e
        except AttributeError as e:
            raise ToolLoadError(
                f"Function '{function_name}' not found in module '{module_name}'"
            ) from e
        
        if not callable(function):
            raise ToolLoadError(
                f"'{module_name}.{function_name}' is not callable"
            )
        
        # Get or generate JSON schema
        json_schema = config.get('schema')
        if json_schema is None:
            # Try to auto-generate schema from function signature
            json_schema = self._generate_schema_from_function(function)
        
        # Check if function takes context
        takes_ctx = config.get('takes_ctx', False)
        
        # Register the tool
        self.registry.register_direct_tool(
            name=name,
            function=function,
            description=description,
            json_schema=json_schema,
            takes_ctx=takes_ctx,
            category=category,
            tags=tags,
        )
    
    def _load_mcp_tool(
        self,
        name: str,
        config: Dict[str, Any],
        description: str,
        category: Optional[str],
        tags: List[str]
    ) -> None:
        """Load an MCP tool from configuration."""
        mcp_server = config.get('mcp_server')
        mcp_tool_name = config.get('mcp_tool_name', name)
        
        if not mcp_server:
            raise ToolLoadError(f"MCP tool '{name}' must specify 'mcp_server'")
        
        # Register the MCP tool
        self.registry.register_mcp_tool(
            name=name,
            mcp_server=mcp_server,
            mcp_tool_name=mcp_tool_name,
            description=description,
            category=category,
            tags=tags,
        )
    
    def _generate_schema_from_function(self, function: Callable) -> Dict[str, Any]:
        """
        Auto-generate a JSON schema from function signature.
        
        Note: This is a basic implementation. For production use,
        consider using pydantic or similar for better type inference.
        """
        sig = inspect.signature(function)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # Skip self, cls, and context parameters
            if param_name in ('self', 'cls', 'ctx', 'context'):
                continue
            
            param_schema = {'type': 'string'}  # Default type
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                param_schema['type'] = self._python_type_to_json_type(param.annotation)
            
            properties[param_name] = param_schema
            
            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        schema = {
            'type': 'object',
            'properties': properties,
            'additionalProperties': False,
        }
        
        if required:
            schema['required'] = required
        
        return schema
    
    def _python_type_to_json_type(self, python_type: Any) -> str:
        """Convert Python type annotation to JSON schema type."""
        type_mapping = {
            int: 'integer',
            float: 'number',
            str: 'string',
            bool: 'boolean',
            list: 'array',
            dict: 'object',
        }
        
        # Handle Optional types
        if hasattr(python_type, '__origin__'):
            origin = python_type.__origin__
            if origin is list:
                return 'array'
            elif origin is dict:
                return 'object'
        
        # Direct type mapping
        return type_mapping.get(python_type, 'string')
    
    def load_from_directory(
        self,
        directory: str,
        pattern: str = "*.yaml",
        recursive: bool = False
    ) -> int:
        """
        Load tools from all configuration files in a directory.
        
        Args:
            directory: Path to directory containing configuration files
            pattern: Glob pattern for configuration files (default: "*.yaml")
            recursive: Whether to search recursively (default: False)
        
        Returns:
            Total number of tools loaded
        
        Raises:
            ToolLoadError: If directory not found or files cannot be loaded
        """
        path = Path(directory)
        if not path.exists():
            raise ToolLoadError(f"Directory not found: {directory}")
        
        if not path.is_dir():
            raise ToolLoadError(f"Not a directory: {directory}")
        
        # Find all matching files
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))
        
        if not files:
            raise ToolLoadError(
                f"No configuration files found matching '{pattern}' in {directory}"
            )
        
        total_loaded = 0
        errors = []
        
        for file_path in files:
            try:
                if file_path.suffix in ['.yaml', '.yml']:
                    count = self.load_from_yaml(str(file_path))
                elif file_path.suffix == '.json':
                    count = self.load_from_json(str(file_path))
                else:
                    continue
                total_loaded += count
            except Exception as e:
                errors.append(f"Error loading {file_path}: {str(e)}")
        
        if errors:
            error_msg = "\n".join(errors)
            raise ToolLoadError(f"Errors loading tools:\n{error_msg}")
        
        return total_loaded


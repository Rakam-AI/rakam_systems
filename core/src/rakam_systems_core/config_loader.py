"""
Configuration loader for agents.

This module provides functionality to load agent configurations from YAML files
and instantiate agents with the specified settings.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, Callable
from pathlib import Path
import yaml
import importlib
import inspect

from pydantic import BaseModel, Field, create_model

from .config_schema import (
    ConfigFileSchema,
    AgentConfigSchema,
    ToolConfigSchema,
    ModelConfigSchema,
    PromptConfigSchema,
    OutputTypeSchema,
    OutputFieldSchema,
    ToolMode,
)
from .interfaces.tool_registry import ToolRegistry, ToolMetadata
from .interfaces.tool import ToolComponent
from .interfaces.agent import ModelSettings


class ConfigurationLoader:
    """
    Loads agent configurations from YAML files and creates agent instances.
    
    Features:
    - Load complete configuration from YAML
    - Resolve references (tools, prompts)
    - Dynamic module/class loading
    - Tool registry integration
    - Validation using Pydantic schemas
    
    Example:
        >>> loader = ConfigurationLoader()
        >>> config = loader.load_from_yaml("agent_config.yaml")
        >>> agent = loader.create_agent("my_agent", config)
    """
    
    def __init__(self):
        self.config: Optional[ConfigFileSchema] = None
        self.tool_registry: Optional[ToolRegistry] = None
    
    def load_from_yaml(self, config_path: str) -> ConfigFileSchema:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Validated configuration schema
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Validate using Pydantic
        self.config = ConfigFileSchema(**raw_config)
        return self.config
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> ConfigFileSchema:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Validated configuration schema
        """
        self.config = ConfigFileSchema(**config_dict)
        return self.config
    
    def get_tool_registry(self, config: Optional[ConfigFileSchema] = None) -> ToolRegistry:
        """
        Get or create a tool registry from configuration.
        
        Args:
            config: Configuration to use (uses loaded config if None)
            
        Returns:
            ToolRegistry with all configured tools
        """
        if self.tool_registry is not None:
            return self.tool_registry
        
        config = config or self.config
        if config is None:
            raise ValueError("No configuration loaded. Call load_from_yaml or load_from_dict first.")
        
        registry = ToolRegistry()
        
        # Register all tools from config
        for tool_name, tool_config in config.tools.items():
            self._register_tool(registry, tool_config)
        
        self.tool_registry = registry
        return registry
    
    def _register_tool(self, registry: ToolRegistry, tool_config: ToolConfigSchema) -> None:
        """
        Register a single tool in the registry.
        
        Args:
            registry: ToolRegistry to register in
            tool_config: Tool configuration
        """
        if tool_config.type == ToolMode.DIRECT:
            # Load the function dynamically
            function = self._load_function(tool_config.module, tool_config.function)
            
            # Register with registry
            registry.register_direct_tool(
                name=tool_config.name,
                function=function,
                description=tool_config.description,
                json_schema=tool_config.json_schema or self._generate_schema(function),
                takes_ctx=tool_config.takes_ctx,
                category=tool_config.category,
                tags=tool_config.tags,
            )
        
        elif tool_config.type == ToolMode.MCP:
            # Register MCP tool
            registry.register_mcp_tool(
                name=tool_config.name,
                mcp_server=tool_config.mcp_server,
                mcp_tool_name=tool_config.mcp_tool_name or tool_config.name,
                description=tool_config.description,
                category=tool_config.category,
                tags=tool_config.tags,
            )
    
    def _load_function(self, module_path: str, function_name: str) -> Callable:
        """
        Dynamically load a function from a module.
        
        Args:
            module_path: Python module path (e.g., 'myapp.tools')
            function_name: Function name to load
            
        Returns:
            The loaded function
        """
        try:
            module = importlib.import_module(module_path)
            function = getattr(module, function_name)
            
            if not callable(function):
                raise ValueError(f"{function_name} in {module_path} is not callable")
            
            return function
        except ImportError as e:
            raise ImportError(f"Failed to import module {module_path}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Function {function_name} not found in {module_path}: {e}")
    
    def _load_class(self, class_path: str) -> Type:
        """
        Dynamically load a class from a module.
        
        Args:
            class_path: Fully qualified class path (e.g., 'myapp.models.MyClass')
            
        Returns:
            The loaded class
        """
        parts = class_path.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid class path: {class_path}")
        
        module_path, class_name = parts
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            
            if not isinstance(cls, type):
                raise ValueError(f"{class_name} in {module_path} is not a class")
            
            return cls
        except ImportError as e:
            raise ImportError(f"Failed to import module {module_path}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Class {class_name} not found in {module_path}: {e}")
    
    def _generate_schema(self, function: Callable) -> Dict[str, Any]:
        """
        Generate a basic JSON schema from function signature.
        
        Args:
            function: Function to generate schema for
            
        Returns:
            JSON schema dictionary
        """
        sig = inspect.signature(function)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # Skip self, cls, context parameters
            if param_name in ['self', 'cls', 'ctx', 'context']:
                continue
            
            # Basic type inference
            param_type = "string"  # default
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in (int, float):
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation in (list, List):
                    param_type = "array"
                elif param.annotation in (dict, Dict):
                    param_type = "object"
            
            properties[param_name] = {"type": param_type}
            
            # Required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    
    def _create_output_type_from_schema(self, output_schema: OutputTypeSchema) -> Type[BaseModel]:
        """
        Dynamically create a Pydantic model from an OutputTypeSchema.
        
        Args:
            output_schema: The output type schema from YAML configuration
            
        Returns:
            A dynamically created Pydantic model class
        
        Example YAML that creates the model:
            output_type:
              name: "SQLAgentOutput"
              description: "Output for SQL agent"
              fields:
                answer:
                  type: str
                  description: "The answer"
                sql_query:
                  type: str
                  description: "The SQL query"
                  default: ""
        """
        # Map string type names to Python types
        type_mapping = {
            'str': str,
            'string': str,
            'int': int,
            'integer': int,
            'float': float,
            'number': float,
            'bool': bool,
            'boolean': bool,
            'list': list,
            'array': list,
            'dict': dict,
            'object': dict,
        }
        
        # Build field definitions for create_model
        field_definitions = {}
        
        for field_name, field_config in output_schema.fields.items():
            # Get the Python type
            python_type = type_mapping.get(field_config.type.lower(), str)
            
            # Determine default value
            if field_config.default is not None:
                default_value = field_config.default
            elif field_config.default_factory:
                # Handle default_factory for mutable types
                if field_config.default_factory.lower() in ('list', 'array', '[]'):
                    default_value = Field(default_factory=list, description=field_config.description)
                    field_definitions[field_name] = (python_type, default_value)
                    continue
                elif field_config.default_factory.lower() in ('dict', 'object', '{}'):
                    default_value = Field(default_factory=dict, description=field_config.description)
                    field_definitions[field_name] = (python_type, default_value)
                    continue
                else:
                    default_value = ...  # Required field
            elif not field_config.required:
                # Optional field with None default
                default_value = None
            else:
                # Required field
                default_value = ...
            
            # Create Field with description
            if default_value is ...:
                field_definitions[field_name] = (
                    python_type, 
                    Field(description=field_config.description)
                )
            else:
                field_definitions[field_name] = (
                    python_type, 
                    Field(default=default_value, description=field_config.description)
                )
        
        # Create the dynamic model
        dynamic_model = create_model(
            output_schema.name,
            __doc__=output_schema.description or f"Dynamically generated output model: {output_schema.name}",
            **field_definitions
        )
        
        return dynamic_model
    
    def _resolve_output_type(self, output_type_config: Optional[str | OutputTypeSchema]) -> Optional[Type]:
        """
        Resolve output_type configuration to a Python class.
        
        Args:
            output_type_config: Either a string class path or an OutputTypeSchema
            
        Returns:
            A Pydantic model class or None
        """
        if output_type_config is None:
            return None
        
        if isinstance(output_type_config, str):
            # It's a class path, load it
            return self._load_class(output_type_config)
        elif isinstance(output_type_config, OutputTypeSchema):
            # It's an inline schema, create the model dynamically
            return self._create_output_type_from_schema(output_type_config)
        else:
            raise ValueError(f"Invalid output_type configuration: {type(output_type_config)}")
    
    def resolve_prompt_config(
        self,
        prompt_ref: str | PromptConfigSchema,
        config: Optional[ConfigFileSchema] = None
    ) -> PromptConfigSchema:
        """
        Resolve a prompt configuration reference.
        
        Args:
            prompt_ref: Prompt name or full configuration
            config: Configuration to use (uses loaded config if None)
            
        Returns:
            Resolved prompt configuration
        """
        if isinstance(prompt_ref, PromptConfigSchema):
            return prompt_ref
        
        config = config or self.config
        if config is None:
            raise ValueError("No configuration loaded")
        
        if prompt_ref not in config.prompts:
            raise ValueError(f"Prompt '{prompt_ref}' not found in configuration")
        
        return config.prompts[prompt_ref]
    
    def resolve_tools(
        self,
        tool_refs: List[str | ToolConfigSchema],
        config: Optional[ConfigFileSchema] = None
    ) -> List[ToolConfigSchema]:
        """
        Resolve tool configuration references.
        
        Args:
            tool_refs: List of tool names or full configurations
            config: Configuration to use (uses loaded config if None)
            
        Returns:
            List of resolved tool configurations
        """
        config = config or self.config
        if config is None:
            raise ValueError("No configuration loaded")
        
        resolved = []
        for ref in tool_refs:
            if isinstance(ref, ToolConfigSchema):
                resolved.append(ref)
            elif isinstance(ref, str):
                if ref not in config.tools:
                    raise ValueError(f"Tool '{ref}' not found in configuration")
                resolved.append(config.tools[ref])
            else:
                raise ValueError(f"Invalid tool reference: {ref}")
        
        return resolved
    
    def create_agent(
        self,
        agent_name: str,
        config: Optional[ConfigFileSchema] = None,
        agent_class: Optional[Type] = None,
    ) -> Any:
        """
        Create an agent instance from configuration.
        
        Args:
            agent_name: Name of agent in configuration
            config: Configuration to use (uses loaded config if None)
            agent_class: Agent class to instantiate (imports BaseAgent if None)
            
        Returns:
            Instantiated agent
        """
        config = config or self.config
        if config is None:
            raise ValueError("No configuration loaded. Call load_from_yaml or load_from_dict first.")
        
        if agent_name not in config.agents:
            raise ValueError(f"Agent '{agent_name}' not found in configuration")
        
        agent_config = config.agents[agent_name]
        
        # Import agent class if not provided
        if agent_class is None:
            from rakam_system_agent.components import BaseAgent
            agent_class = BaseAgent
        
        # Resolve prompt config
        prompt_config = self.resolve_prompt_config(agent_config.prompt_config, config)
        
        # Resolve tools
        tool_configs = self.resolve_tools(agent_config.tools, config)
        
        # Create tool registry with resolved tools
        registry = ToolRegistry()
        for tool_config in tool_configs:
            self._register_tool(registry, tool_config)
        
        # Load deps_type if specified
        deps_type = None
        if agent_config.deps_type:
            deps_type = self._load_class(agent_config.deps_type)
        
        # Resolve output_type (can be string class path or inline schema)
        output_type = self._resolve_output_type(agent_config.output_type)
        
        # Convert model config to ModelSettings
        model_settings = ModelSettings(
            parallel_tool_calls=agent_config.llm_config.parallel_tool_calls,
            temperature=agent_config.llm_config.temperature,
            max_tokens=agent_config.llm_config.max_tokens,
            **agent_config.llm_config.extra_settings,
        )
        
        # Create agent configuration dict
        config_dict = {
            "model": agent_config.llm_config.model,
            "stateful": agent_config.stateful,
            "system_prompt": prompt_config.system_prompt,
            **agent_config.metadata,
        }
        
        # Instantiate agent
        agent = agent_class(
            name=agent_config.name,
            config=config_dict,
            model=agent_config.llm_config.model,
            deps_type=deps_type,
            output_type=output_type,
            system_prompt=prompt_config.system_prompt,
            tool_registry=registry,
        )
        
        # Store tracking configuration
        agent._tracking_enabled = agent_config.enable_tracking
        agent._tracking_output_dir = agent_config.tracking_output_dir
        
        return agent
    
    def create_all_agents(
        self,
        config: Optional[ConfigFileSchema] = None,
        agent_class: Optional[Type] = None,
    ) -> Dict[str, Any]:
        """
        Create all agents from configuration.
        
        Args:
            config: Configuration to use (uses loaded config if None)
            agent_class: Agent class to instantiate (imports BaseAgent if None)
            
        Returns:
            Dictionary mapping agent names to instances
        """
        config = config or self.config
        if config is None:
            raise ValueError("No configuration loaded")
        
        agents = {}
        for agent_name in config.agents:
            agents[agent_name] = self.create_agent(agent_name, config, agent_class)
        
        return agents
    
    def validate_config(self, config_path: Optional[str] = None) -> tuple[bool, List[str]]:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to config file (uses loaded config if None)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            if config_path:
                config = self.load_from_yaml(config_path)
            else:
                config = self.config
                if config is None:
                    return False, ["No configuration loaded"]
            
            # Validate agent references
            for agent_name, agent_config in config.agents.items():
                # Check prompt reference
                try:
                    self.resolve_prompt_config(agent_config.prompt_config, config)
                except ValueError as e:
                    errors.append(f"Agent '{agent_name}': {e}")
                
                # Check tool references
                try:
                    self.resolve_tools(agent_config.tools, config)
                except ValueError as e:
                    errors.append(f"Agent '{agent_name}': {e}")
                
                # Check deps_type if specified
                if agent_config.deps_type:
                    try:
                        self._load_class(agent_config.deps_type)
                    except (ImportError, AttributeError) as e:
                        errors.append(f"Agent '{agent_name}' deps_type: {e}")
                
                # Check output_type if specified
                if agent_config.output_type:
                    try:
                        self._resolve_output_type(agent_config.output_type)
                    except (ImportError, AttributeError, ValueError) as e:
                        errors.append(f"Agent '{agent_name}' output_type: {e}")
            
            # Validate tools can be loaded
            for tool_name, tool_config in config.tools.items():
                if tool_config.type == ToolMode.DIRECT:
                    try:
                        self._load_function(tool_config.module, tool_config.function)
                    except (ImportError, AttributeError) as e:
                        errors.append(f"Tool '{tool_name}': {e}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Configuration error: {e}"]


from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
import inspect
from ..base import BaseComponent


class ToolComponent(BaseComponent, ABC):
    """
    Represents a callable external or internal tool, compatible with Pydantic AI.
    
    This is the base class for all tools in the system. Tools can be functions
    or callable objects that can be invoked by agents.
    
    Attributes:
        name: Unique name for the tool
        description: Human-readable description
        function: The callable function (defaults to self.run)
        json_schema: JSON schema for tool parameters
        takes_ctx: Whether the tool takes context as first argument
        is_async: Whether the function is async
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        takes_ctx: bool = False,
    ) -> None:
        """
        Initialize a ToolComponent.
        
        Args:
            name: Unique name for the tool
            config: Optional configuration dictionary
            description: Human-readable description of what the tool does
            json_schema: JSON schema defining the tool's parameters
            takes_ctx: Whether the tool takes context as first argument
        """
        super().__init__(name, config)
        self.description = description or f"Tool: {name}"
        self.json_schema = json_schema or self._generate_default_schema()
        self.takes_ctx = takes_ctx
        
        # Set function to the run method for Pydantic AI compatibility
        # Check if run method is async
        self.function = self.run
        self.is_async = inspect.iscoroutinefunction(self.run)
    
    def _generate_default_schema(self) -> Dict[str, Any]:
        """
        Generate a default JSON schema for the tool.
        Subclasses can override this to provide custom schemas.
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Input query or parameter for the tool"
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }
    
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the primary operation for the tool.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError
    
    @classmethod
    def from_function(
        cls,
        function: Callable[..., Any],
        name: str,
        description: str,
        json_schema: Dict[str, Any],
        takes_ctx: bool = False,
    ) -> "ToolComponent":
        """
        Create a ToolComponent from a function (Pydantic AI compatible).
        
        This factory method allows creating tool instances from standalone functions.
        
        Args:
            function: The callable function
            name: Unique name for the tool
            description: Human-readable description
            json_schema: JSON schema for parameters
            takes_ctx: Whether the tool takes context as first argument
            
        Returns:
            FunctionToolComponent instance wrapping the function
        """
        return FunctionToolComponent(
            function=function,
            name=name,
            description=description,
            json_schema=json_schema,
            takes_ctx=takes_ctx,
        )
    
    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """
        Async call for the tool.
        Automatically handles both sync and async run methods.
        """
        if self.is_async:
            return await self.run(*args, **kwargs)
        else:
            return self.run(*args, **kwargs)


class FunctionToolComponent(ToolComponent):
    """
    A ToolComponent that wraps a standalone function.
    
    This is used internally by the from_function factory method to wrap
    plain functions as ToolComponent instances.
    """
    
    def __init__(
        self,
        function: Callable[..., Any],
        name: str,
        description: str,
        json_schema: Dict[str, Any],
        takes_ctx: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Store the function before calling super().__init__
        self._wrapped_function = function
        
        # Initialize the parent with the description and schema
        super().__init__(
            name=name,
            config=config,
            description=description,
            json_schema=json_schema,
            takes_ctx=takes_ctx,
        )
        
        # Override function and is_async based on the wrapped function
        self.function = function
        self.is_async = inspect.iscoroutinefunction(function)
    
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the wrapped function."""
        return self._wrapped_function(*args, **kwargs)
    
    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Async call for the wrapped function."""
        if self.is_async:
            return await self._wrapped_function(*args, **kwargs)
        else:
            return self._wrapped_function(*args, **kwargs)

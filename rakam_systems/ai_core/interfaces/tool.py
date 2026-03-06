from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Awaitable
import inspect
from ..base import BaseComponent

class ToolComponent(BaseComponent, ABC):
    """Represents a callable external or internal tool."""
    @abstractmethod
    def run(self, query: str) -> Any:
        raise NotImplementedError

class Tool:
    """Tool wrapper compatible with Pydantic AI's Tool.from_schema pattern."""
    
    def __init__(
        self,
        function: Callable[..., Any],
        name: str,
        description: str,
        json_schema: Dict[str, Any],
        takes_ctx: bool = False,
    ) -> None:
        self.function = function
        self.name = name
        self.description = description
        self.json_schema = json_schema
        self.takes_ctx = takes_ctx
        self.is_async = inspect.iscoroutinefunction(function)
    
    @classmethod
    def from_schema(
        cls,
        function: Callable[..., Any],
        name: str,
        description: str,
        json_schema: Dict[str, Any],
        takes_ctx: bool = False,
    ) -> "Tool":
        """Create a Tool from a schema definition (Pydantic AI compatible)."""
        return cls(
            function=function,
            name=name,
            description=description,
            json_schema=json_schema,
            takes_ctx=takes_ctx,
        )
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function."""
        return self.function(*args, **kwargs)
    
    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Async call for the underlying function."""
        if self.is_async:
            return await self.function(*args, **kwargs)
        else:
            return self.function(*args, **kwargs)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from ..base import BaseComponent

class ToolComponent(BaseComponent, ABC):
    """Represents a callable external or internal tool."""
    @abstractmethod
    def run(self, query: str) -> Any:
        raise NotImplementedError

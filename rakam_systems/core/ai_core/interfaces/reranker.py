from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from ..base import BaseComponent

class Reranker(BaseComponent, ABC):
    @abstractmethod
    def run(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder documents by relevance and return a new list."""
        raise NotImplementedError

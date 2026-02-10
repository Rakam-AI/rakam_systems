from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from ..base import BaseComponent

class Indexer(BaseComponent, ABC):
    @abstractmethod
    def run(self, documents: List[str], embeddings: List[List[float]]) -> Any:
        """Index documents + embeddings into a backing store."""
        raise NotImplementedError

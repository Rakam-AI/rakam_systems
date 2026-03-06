from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from ..base import BaseComponent

class Chunker(BaseComponent, ABC):
    @abstractmethod
    def run(self, documents: List[str]) -> List[str]:
        """Split documents into smaller chunks.
        Keep pure string IO to avoid extra dependencies."""
        raise NotImplementedError

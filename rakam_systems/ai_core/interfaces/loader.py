from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from ..base import BaseComponent

class Loader(BaseComponent, ABC):
    @abstractmethod
    def run(self, source: str) -> List[str]:
        """Load raw documents from a source (path, URL, id, etc.)."""
        raise NotImplementedError

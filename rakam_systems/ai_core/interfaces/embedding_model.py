from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from ..base import BaseComponent

class EmbeddingModel(BaseComponent, ABC):
    @abstractmethod
    def run(self, texts: List[str]) -> List[List[float]]:
        """Return one vector per input text."""
        raise NotImplementedError

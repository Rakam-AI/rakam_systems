from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from ..base import BaseComponent

class Retriever(BaseComponent, ABC):
    @abstractmethod
    def run(self, query: str) -> List[Dict[str, Any]]:
        """Return a list of candidate hits with metadata.
        Output schema is intentionally loose to stay dependency‑free."""
        raise NotImplementedError

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..base import BaseComponent


class VectorStore(BaseComponent, ABC):
    @abstractmethod
    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def query(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """
        Convenience default so subclasses don't *have* to implement run().
        If called with a 'vector' (or first positional arg), proxies to query().
        """
        vector = kwargs.get("vector")
        if vector is None and args:
            vector = args[0]
        top_k = kwargs.get("top_k", 5)
        if vector is None:
            raise NotImplementedError(
                "VectorStore.run expects a 'vector' argument or an override."
            )
        return self.query(vector, top_k=top_k)

    def count(self) -> Optional[int]:
        return None


# compatibility alias
VectorStoreComponent = VectorStore
__all__ = ["VectorStore", "VectorStoreComponent"]

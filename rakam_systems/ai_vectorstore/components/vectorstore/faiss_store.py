from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from ai_core.interfaces.vectorstore import VectorStore

class FaissStore(VectorStore):
    """Name-only placeholder. No FAISS dependency.
    Subclass this and implement the methods with a real backend.
    """
    def __init__(self, name: str = "faiss_store", config=None) -> None:
        super().__init__(name, config)

    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> Any:
        raise NotImplementedError("FaissStore.add requires a concrete backend implementation")

    def query(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError("FaissStore.query requires a concrete backend implementation")

    def count(self) -> Optional[int]:
        return None

from __future__ import annotations
from typing import Any, Dict, List
from ai_core.interfaces.indexer import Indexer

class SimpleIndexer(Indexer):
    """Thin wrapper around a VectorStore (to be injected by caller)."""
    def __init__(self, name: str, vectorstore, config=None) -> None:
        super().__init__(name, config)
        self.vectorstore = vectorstore

    def run(self, documents: List[str], embeddings: List[List[float]]) -> Any:
        metadatas = [{"doc_index": i} for i in range(len(documents))]
        return self.vectorstore.add(embeddings, metadatas)

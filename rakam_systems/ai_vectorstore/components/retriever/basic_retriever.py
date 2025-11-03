from __future__ import annotations
from typing import Any, Dict, List
from ai_core.interfaces.retriever import Retriever

class BasicRetriever(Retriever):
    """Minimal retriever that depends on an injected VectorStore and query encoder."""
    def __init__(self, name: str, vectorstore, encoder, config=None) -> None:
        super().__init__(name, config)
        self.vectorstore = vectorstore
        self.encoder = encoder  # callable: str -> List[float]

    def run(self, query: str) -> List[Dict[str, Any]]:
        vec = self.encoder(query)
        return self.vectorstore.query(vec, top_k=int(self.config.get("top_k", 5)))

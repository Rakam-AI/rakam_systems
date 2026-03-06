from __future__ import annotations
from typing import Any, Dict, List
from ai_core.interfaces.reranker import Reranker

class ModelReranker(Reranker):
    """Abstract reranker placeholder.
    Implement score computation in a concrete subclass.
    """
    def run(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError("ModelReranker.run must be implemented by a concrete subclass")

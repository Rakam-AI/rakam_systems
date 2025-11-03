from __future__ import annotations
from typing import List
from ai_core.interfaces.embedding_model import EmbeddingModel

class OpenAIEmbeddings(EmbeddingModel):
    """Name-only placeholder. No SDKs are imported.
    Implement network calls in a subclass or adapt to a local model.
    """
    def run(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("OpenAIEmbeddings.run must be implemented with a real embedding backend")

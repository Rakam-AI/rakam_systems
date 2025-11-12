from __future__ import annotations
from typing import List
from rakam_systems.ai_core.interfaces.chunker import Chunker

class SimpleChunker(Chunker):
    """Simple paragraph splitter example; override in production."""
    def run(self, documents: List[str]) -> List[str]:
        # Default behavior: return input unchanged to keep it abstract.
        # Subclasses may implement naive splitting (e.g., by \n\n) or token-count rules.
        return list(documents)

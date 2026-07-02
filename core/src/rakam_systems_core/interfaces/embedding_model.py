from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import List
from ..base import BaseComponent

class EmbeddingModel(BaseComponent, ABC):
    @abstractmethod
    def run(self, texts: List[str]) -> List[List[float]]:
        """Return one vector per input text (synchronous)."""
        raise NotImplementedError

    async def arun(self, texts: List[str]) -> List[List[float]]:
        """Async variant of :meth:`run`.

        Default implementation offloads the sync ``run`` to a thread so callers
        already inside an event loop (e.g. async FastAPI services) don't block
        it. Async-native backends override this to await directly — see the
        gateway embedder, which awaits pydantic-ai's async ``embed_documents``.
        """
        return await asyncio.get_running_loop().run_in_executor(None, self.run, texts)

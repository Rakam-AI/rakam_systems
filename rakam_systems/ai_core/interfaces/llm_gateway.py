from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional
from ..base import BaseComponent

class LLMGateway(BaseComponent, ABC):
    @abstractmethod
    def run(self, prompt: str, **kwargs: Any) -> str:
        """Synchronous text completion."""
        raise NotImplementedError

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Optional token/segment stream; default to single-yield for compatibility."""
        yield self.run(prompt, **kwargs)

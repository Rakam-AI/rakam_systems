from __future__ import annotations
from typing import Any, Iterator
from ai_core.interfaces.llm_gateway import LLMGateway

class OpenAIGateway(LLMGateway):
    """Naming placeholder. This class does not import any SDKs.
    Implement network calls in a subclass or replace at runtime.
    """

    def run(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError("OpenAIGateway.run must be implemented with a real client")  # noqa: E501

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        # Default to non-streaming fallback
        yield self.run(prompt, **kwargs)

from __future__ import annotations
from typing import Iterator
from ai_core.interfaces.agent import AgentComponent, AgentInput, AgentOutput

class BaseAgent(AgentComponent):
    """A convenient partial implementation of AgentComponent.
    Subclasses only need to implement `infer()`.
    """

    def infer(self, input_data: AgentInput) -> AgentOutput:
        """Override to implement non-streaming inference."""
        raise NotImplementedError

    def run(self, input_data: AgentInput) -> AgentOutput:
        return self.infer(input_data)

    def stream(self, input_data: AgentInput) -> Iterator[str]:
        # Default: produce a single chunk = final text
        out = self.infer(input_data)
        yield out.output_text

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional
from ..base import BaseComponent

class AgentInput:
    """Simple DTO for agent inputs (no external deps)."""
    def __init__(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.input_text = input_text
        self.context = context or {}

class AgentOutput:
    """Simple DTO for agent outputs."""
    def __init__(self, output_text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.output_text = output_text
        self.metadata = metadata or {}

class AgentComponent(BaseComponent, ABC):
    """Abstract agent interface with optional streaming."""
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self.stateful: bool = bool((config or {}).get("stateful", False))

    @abstractmethod
    def run(self, input_data: AgentInput) -> AgentOutput:
        raise NotImplementedError

    def stream(self, input_data: AgentInput) -> Iterator[str]:
        """Optional streaming interface: by default yields the final text once."""
        out = self.run(input_data)
        yield out.output_text

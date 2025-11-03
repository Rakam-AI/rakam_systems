from __future__ import annotations

from typing import Dict, Any
from ..base import BaseComponent


class MCPServer(BaseComponent):
    """Micro, dependency‑free MCP‑like registry.
    Not a real MCP wire implementation—just a local message router so
    components can "send" messages during tests.
    """

    def __init__(self, name: str = "mcp_server", config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self._registry: Dict[str, BaseComponent] = {}

    def register_component(self, component: BaseComponent) -> None:
        self._registry[component.name] = component

    def send_message(self, sender: str, receiver: str, message: Dict[str, Any]) -> Any:
        if receiver not in self._registry:
            raise KeyError(f"Receiver '{receiver}' not registered")
        target = self._registry[receiver]
        handler = getattr(target, "handle_message", None)
        if callable(handler):
            return handler(sender=sender, message=message)
        # Default: call run if no handler provided
        return target.run(message)

    def run(self, *args, **kwargs):
        # Server has no primary run path; return registry snapshot.
        return sorted(self._registry.keys())

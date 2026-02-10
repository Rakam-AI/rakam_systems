from __future__ import annotations
from typing import Any
# from rakam_systems_core.ai_core.interfaces.tool import ToolComponent
from rakam_systems_core.ai_core.interfaces.tool import ToolComponent


class SearchTool(ToolComponent):
    """Abstract search tool placeholder.
    Implementors should call an external search API or local index.
    """

    def run(self, query: str) -> Any:
        raise NotImplementedError(
            "SearchTool.run must be implemented by a concrete subclass")

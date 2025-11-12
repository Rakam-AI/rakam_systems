from __future__ import annotations
from typing import List
from rakam_systems.ai_core.interfaces.loader import Loader

class FileLoader(Loader):
    """Dependency-free file loader stub.
    Reads text files when run() is implemented in a concrete subclass.
    """
    def run(self, source: str) -> List[str]:
        raise NotImplementedError("FileLoader.run must read from filesystem or other sources")

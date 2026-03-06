from __future__ import annotations
from typing import Any

def get_logger(name: str = "ai") -> Any:
    """Return a tiny print-based logger to avoid dependencies."""
    class _L:
        def info(self, msg: str, **kw): print(f"[INFO] {name}: {msg}", kw if kw else "")
        def warning(self, msg: str, **kw): print(f"[WARN] {name}: {msg}", kw if kw else "")
        def error(self, msg: str, **kw): print(f"[ERROR] {name}: {msg}", kw if kw else "")
        def debug(self, msg: str, **kw): print(f"[DEBUG] {name}: {msg}", kw if kw else "")
    return _L()

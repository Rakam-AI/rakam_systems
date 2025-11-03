from __future__ import annotations
from typing import Dict

_METRICS: Dict[str, float] = {}

def record_metric(name: str, value: float) -> None:
    _METRICS[name] = value

def get_metric(name: str) -> float | None:
    return _METRICS.get(name)

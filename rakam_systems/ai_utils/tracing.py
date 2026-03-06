from __future__ import annotations
# Intentionally minimal; extend with real tracing later.

def init_tracing(service_name: str = "ai_system") -> None:
    print(f"Tracing initialized for {service_name} (noop)")

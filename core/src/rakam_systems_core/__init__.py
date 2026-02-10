"""
Rakam System Core - Core abstractions and data structures for the AI system.
"""

from .ai_core.vs_core import Node, NodeMetadata, VSFile
from .ai_core import (
    AgentComponent,
    AgentInput,
    AgentOutput,
    ModelSettings,
    ToolComponent,
    ConfigurationLoader,
    TrackingManager,
    TrackingMixin,
    track_method,
    get_tracking_manager,
)

__all__ = [
    # Core data structures
    "Node",
    "NodeMetadata",
    "VSFile",
    # Core interfaces
    "AgentComponent",
    "AgentInput",
    "AgentOutput",
    "ModelSettings",
    "ToolComponent",
    # Configuration
    "ConfigurationLoader",
    # Tracking
    "TrackingManager",
    "TrackingMixin",
    "track_method",
    "get_tracking_manager",
]


def hello() -> str:
    return "Hello from rakam-systems-core!"

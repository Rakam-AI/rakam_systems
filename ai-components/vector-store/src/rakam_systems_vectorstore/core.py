"""
Re-export core data structures from rakam-systems-core for backward compatibility.
These classes are now maintained in rakam_systems_core.ai_core.vs_core
"""

from rakam_systems_core.ai_core.vs_core import Node, NodeMetadata, VSFile

__all__ = ["Node", "NodeMetadata", "VSFile"]

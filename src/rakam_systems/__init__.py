"""Rakam Systems — modular AI framework."""

import sys

import rakam_systems_agent as agent
import rakam_systems_cli as cli
import rakam_systems_core as core
import rakam_systems_tools as tools
import rakam_systems_vectorstore as vectorstore

# Register aliases so `from rakam_systems.agent import X` resolves correctly.
sys.modules.update(
    {
        "rakam_systems.agent": agent,
        "rakam_systems.agents": agent,
        "rakam_systems.cli": cli,
        "rakam_systems.core": core,
        "rakam_systems.tools": tools,
        "rakam_systems.vectorstore": vectorstore,
    }
)

__all__ = ["core", "cli", "agent", "vectorstore", "tools"]

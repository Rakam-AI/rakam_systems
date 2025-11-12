from __future__ import annotations

import yaml
from pathlib import Path
from typing import Iterator, Any, Dict, List

from rakam_systems.ai_core.interfaces.agent import AgentInput, AgentOutput
from rakam_systems.ai_agents.components.base_agent import BaseAgent
from rakam_systems.ai_vectorstore.components.retriever.basic_retriever import BasicRetriever
from rakam_systems.ai_core.interfaces.vectorstore import VectorStore


# ---------------------------------------------------------------------
# Helper: load YAML safely
# ---------------------------------------------------------------------
def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------
# In-memory vector store (for demo)
# ---------------------------------------------------------------------
class InMemoryVectorStore(VectorStore):
    def __init__(self, name: str = "mem_vs", config=None) -> None:
        super().__init__(name, config)
        self._vecs: List[List[float]] = []
        self._metas: List[Dict[str, Any]] = []

    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> Any:
        self._vecs.extend(vectors)
        self._metas.extend(metadatas)
        return len(self._vecs)

    def query(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        def l2(a, b):
            return sum((x - y) ** 2 for x, y in zip(a, b))

        ranked = sorted(zip(self._vecs, self._metas), key=lambda t: l2(t[0], vector))
        return [m for _, m in ranked[:top_k]]

    def count(self) -> int:
        return len(self._vecs)


# ---------------------------------------------------------------------
# Simple deterministic encoder
# ---------------------------------------------------------------------
def toy_encoder(text: str) -> List[float]:
    out = [0.0] * 8
    for i, ch in enumerate(text.encode("utf-8")):
        out[i % 8] += (ch % 13) / 13.0
    return out


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------
class EchoAgent(BaseAgent):
    def infer(self, input_data: AgentInput) -> AgentOutput:
        prefix = input_data.context.get("prefix", "ECHO")
        return AgentOutput(
            f"{prefix}: {input_data.input_text}",
            metadata={"len": len(input_data.input_text)},
        )


# ---------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Load YAML configs
    cfg_dir = Path(__file__).parent / "configs"
    agent_cfg = load_config(cfg_dir / "agent_config.yaml")
    vector_cfg = load_config(cfg_dir / "vectorstore_config.yaml")

    print("Loaded configs:")
    print(" - agent:", agent_cfg)
    print(" - vectorstore:", vector_cfg)

    # Build components
    vs = InMemoryVectorStore("mem")
    retriever = BasicRetriever(
        "retriever", vectorstore=vs, encoder=toy_encoder, config=vector_cfg
    )
    agent = EchoAgent("echo", config=agent_cfg)

    # Seed data
    vs.add(
        [toy_encoder("hello"), toy_encoder("world")],
        metadatas=[{"text": "hello"}, {"text": "world"}],
    )
    print("Vector count:", vs.count())

    # Run agent
    inp = AgentInput(
        "test message", context={"prefix": agent_cfg.get("prefix", "AGENT")}
    )
    out = agent.run(inp)
    print("Agent output:", out.output_text)

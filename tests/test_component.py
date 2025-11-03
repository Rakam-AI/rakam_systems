import pytest
from typing import List, Dict, Any
from pydantic import BaseModel
from rakam_systems.ai_core.interfaces.agent import (
    AgentInput,
    AgentOutput,
    AgentComponent,
)
from rakam_systems.ai_core.interfaces.tool import ToolComponent
from rakam_systems.ai_core.interfaces.loader import Loader
from rakam_systems.ai_core.interfaces.chunker import Chunker
from rakam_systems.ai_core.interfaces.indexer import Indexer
from rakam_systems.ai_core.interfaces.vectorstore import VectorStore
from rakam_systems.ai_core.interfaces.retriever import Retriever
from rakam_systems.ai_core.interfaces.reranker import Reranker
from rakam_systems.ai_core.interfaces.llm_gateway import LLMGateway
from rakam_systems.ai_core.interfaces.embedding_model import EmbeddingModel
from rakam_systems.ai_core.mcp.mcp_server import MCPServer


# ---------------------------------------------------------------------
# Mock implementations for testing
# ---------------------------------------------------------------------


class DummyTool(ToolComponent):
    def run(self, query: str) -> str:
        return f"TOOL_RESULT: {query}"


class DummyAgent(AgentComponent):
    def run(self, input_data: AgentInput) -> AgentOutput:
        return AgentOutput(output_text=f"AGENT_OUTPUT: {input_data.input_text}")

    def stream(self, input_data: AgentInput):
        yield f"STREAM_START: {input_data.input_text}"
        yield f"STREAM_END: {input_data.input_text}"


class DummyMCP(MCPServer):
    def __init__(self, name: str):
        super().__init__(name)
        self.registered = []

    def register_component(self, component):
        self.registered.append(component.name)

    def send_message(self, sender: str, receiver: str, message: Dict[str, Any]):
        return {"sender": sender, "receiver": receiver, "message": message}


class DummyLoader(Loader):
    def run(self, source: str) -> List[str]:
        return [f"Loaded from {source}"]


class DummyChunker(Chunker):
    def run(self, documents: List[str]) -> List[str]:
        return [doc[:5] for doc in documents]


class DummyIndexer(Indexer):
    def run(self, documents: List[str], embeddings: List[List[float]]):
        return {"indexed": len(documents)}


class DummyVectorStore(VectorStore):
    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        return {"added": len(vectors)}

    def query(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        return [{"id": i, "score": 1.0 / (i + 1)} for i in range(top_k)]


class DummyRetriever(Retriever):
    def run(self, query: str) -> List[Dict[str, Any]]:
        return [{"text": query, "score": 0.9}]


class DummyReranker(Reranker):
    def run(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(documents, key=lambda x: -x.get("score", 0))


class DummyLLM(LLMGateway):
    def run(self, prompt: str, **kwargs) -> str:
        return f"LLM_REPLY: {prompt}"


class DummyEmbedding(EmbeddingModel):
    def run(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] for t in texts]


# ---------------------------------------------------------------------
# Unit tests for all components
# ---------------------------------------------------------------------


def test_tool_run():
    tool = DummyTool("test_tool")
    result = tool.run("query")
    assert "TOOL_RESULT" in result


def test_agent_run_and_stream():
    agent = DummyAgent("test_agent")
    output = agent.run(AgentInput(input_text="Hello"))
    assert isinstance(output, AgentOutput)
    assert "AGENT_OUTPUT" in output.output_text

    stream_results = list(agent.stream(AgentInput(input_text="Hello")))
    assert len(stream_results) == 2
    assert stream_results[0].startswith("STREAM_START")


def test_mcp_register_and_send():
    mcp = DummyMCP("mcp_test")
    tool = DummyTool("t1")
    mcp.register_component(tool)
    assert tool.name in mcp.registered

    msg = mcp.send_message("A", "B", {"data": 42})
    assert msg["sender"] == "A"
    assert msg["receiver"] == "B"


def test_loader_run():
    loader = DummyLoader("loader")
    docs = loader.run("source.txt")
    assert "Loaded" in docs[0]


def test_chunker_run():
    chunker = DummyChunker("chunker")
    docs = ["document1", "document2"]
    chunks = chunker.run(docs)
    assert all(len(c) <= 5 for c in chunks)


def test_indexer_run():
    indexer = DummyIndexer("indexer")
    result = indexer.run(["doc"], [[0.1, 0.2]])
    assert result["indexed"] == 1


def test_vectorstore_add_and_query():
    store = DummyVectorStore("vectorstore")
    add_result = store.add([[0.1, 0.2]], [{"id": 1}])
    assert add_result["added"] == 1

    query_result = store.query([0.5, 0.5], top_k=3)
    assert len(query_result) == 3
    assert all("score" in r for r in query_result)


def test_retriever_run():
    retriever = DummyRetriever("retriever")
    results = retriever.run("query")
    assert results[0]["score"] == pytest.approx(0.9)


def test_reranker_run():
    reranker = DummyReranker("reranker")
    docs = [{"text": "a", "score": 0.5}, {"text": "b", "score": 0.9}]
    reranked = reranker.run(docs)
    assert reranked[0]["score"] >= reranked[1]["score"]


def test_llm_run():
    llm = DummyLLM("llm")
    reply = llm.run("Hello")
    assert "LLM_REPLY" in reply


def test_embedding_run():
    embed = DummyEmbedding("embedder")
    vecs = embed.run(["abc", "hello"])
    assert all(isinstance(v, list) for v in vecs)
    assert all(isinstance(v[0], float) for v in vecs)


def test_evaluate_method():
    """Ensure BaseComponent.evaluate works generically."""
    tool = DummyTool("tool_eval")
    test_cases = {"run": [{"args": ["test"], "expected": "TOOL_RESULT: test"}]}
    results = tool.evaluate(
        methods=["run"],
        test_cases=test_cases,
        metric_fn=lambda out, exp: 1.0 if out == exp else 0.0,
    )
    assert "run" in results
    assert results["run"][0]["score"] == 1.0
    assert results["run"][0]["success"]

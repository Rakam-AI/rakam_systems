import pytest
import pandas as pd
from typing import Any

from rakam_systems.core import Node, NodeMetadata
from rakam_systems.generation.llm import LLM
from rakam_systems.vector_store import VectorStores
from rakam_systems.generation.agents import ClassifyQuery, RAGGeneration, Agent, Action


# Simple LLM Implementation
class SimpleLLM(LLM):
    def __init__(self, model="gpt-4o", api_key=None):
        self.model = model
        self.api_key = api_key

    def call_llm(self, sys_prompt: str, prompt: str) -> str:
        return f"Generated response for: {prompt}"

    def call_llm_stream(self, sys_prompt: str, prompt: str):
        responses = ["Streamed", " response", " from", " LLM"]
        for response in responses:
            yield response


# Simple Vector Store Implementation
class SimpleVectorStore(VectorStores):
    def __init__(
        self,
        base_index_path="temp_path",
        embedding_model="all-MiniLM-L6-v2",
    ):
        self.stores = {}

    def create_from_nodes(self, store_name: str, nodes: list):
        self.stores[store_name] = nodes

    def search(self, store_name: str, query: str, number: int = 2):
        nodes = self.stores.get(store_name, [])
        return None, nodes[:number]


class ConcreteAgent(Agent):
    def __init__(self, model: str, api_key: str):
        super().__init__(model=model, api_key=api_key)

    def choose_action(self, input: str, state: Any = None) -> Action:
        # Simply return the action based on the input for testing purposes
        return self.actions.get(input)


@pytest.fixture
def simple_llm():
    return SimpleLLM()


@pytest.fixture
def simple_vector_store():
    vector_store = SimpleVectorStore()
    nodes = [
        Node(
            content="This is a mock content 1.",
            metadata=NodeMetadata(
                source_file_uuid="mock_uuid_1",
                position=None,
                custom={"class_name": "mock_class_1"},
            ),
        ),
        Node(
            content="This is a mock content 2.",
            metadata=NodeMetadata(
                source_file_uuid="mock_uuid_2",
                position=None,
                custom={"class_name": "mock_class_2"},
            ),
        ),
    ]
    vector_store.create_from_nodes("query_classification", nodes)
    return vector_store


@pytest.fixture
def trigger_queries():
    return pd.Series(["trigger query 1", "trigger query 2"])


@pytest.fixture
def class_names():
    return pd.Series(["class 1", "class 2"])


@pytest.fixture
def simple_agent(simple_llm):
    return ConcreteAgent(model="gpt-4o", api_key="dummy_api_key")


### Tests for ClassifyQuery


def test_classify_query_initialization(
    trigger_queries, class_names, simple_vector_store, simple_agent
):
    action = ClassifyQuery(
        simple_agent,
        trigger_queries,
        class_names,
        embedding_model="all-MiniLM-L6-v2",
    )

    assert action.agent == simple_agent
    assert action.trigger_queries.equals(trigger_queries)
    assert action.class_names.equals(class_names)
    assert action.embedding_model == "all-MiniLM-L6-v2"
    # assert isinstance(action.vector_store, SimpleVectorStore)


def test_classify_query_execution(
    trigger_queries, class_names, simple_vector_store, simple_agent
):
    action = ClassifyQuery(
        simple_agent,
        trigger_queries,
        class_names,
        embedding_model="all-MiniLM-L6-v2",
    )
    action.vector_store = simple_vector_store

    query = "test query"
    class_name, matched_trigger_query = action.execute(query=query)

    assert class_name == "mock_class_1"
    assert matched_trigger_query == "This is a mock content 1."


### Tests for RAGGeneration


def test_rag_generation_initialization(simple_vector_store, simple_llm, simple_agent):
    action = RAGGeneration(
        simple_agent,
        "System Prompt",
        "Prompt {query}",
        simple_vector_store,
        vs_descriptions={"query_classification": "Mock Description"},
    )

    assert action.agent == simple_agent
    assert action.sys_prompt == "System Prompt"
    assert action.prompt == "Prompt {query}"
    assert action.vector_stores == simple_vector_store
    assert action.vs_descriptions["query_classification"] == "Mock Description"


# def test_rag_generation_non_stream(simple_vector_store, simple_llm, simple_agent):
#     action = RAGGeneration(
#         simple_agent, "System Prompt", "Prompt {query}", simple_vector_store
#     )

#     result = action.execute("test query", stream=False)

#     # assert result == "Generated response for: Prompt test query"


# def test_rag_generation_stream(simple_vector_store, simple_llm, simple_agent):
#     action = RAGGeneration(
#         simple_agent, "System Prompt", "Prompt {query}", simple_vector_store
#     )

#     result = list(action.execute("test query", stream=True))

#     # assert result == ["Streamed", " response", " from", " LLM"]


### Tests for Agent


def test_agent_initialization(simple_llm):
    agent = ConcreteAgent(model="gpt-4o", api_key="dummy_api_key")
    assert agent.llm.model == "gpt-4o"
    assert agent.state == {}
    assert agent.actions == {}


def test_agent_add_action(simple_agent):
    action = ClassifyQuery(simple_agent, pd.Series(["query"]), pd.Series(["class"]))

    simple_agent.add_action("classify_query", action)

    assert "classify_query" in simple_agent.actions
    assert simple_agent.actions["classify_query"] == action


def test_agent_execute_action(simple_agent):
    action = ClassifyQuery(simple_agent, pd.Series(["query"]), pd.Series(["class"]))
    simple_agent.add_action("classify_query", action)

    # Override choose_action to return the correct action
    simple_agent.choose_action = lambda x, state=None: action

    result = simple_agent.execute_action("classify_query", query="test query")

    assert result is not None

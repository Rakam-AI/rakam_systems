import pytest
from rakam_systems.components.agents.mock_agent import MockAgent, MockAction
from rakam_systems.components.agents.actions import (
    TextSearchMetadata, 
    ClassifyQuery, 
    RAGGeneration, 
    GenericLLMResponse
)
import pandas as pd


@pytest.fixture
def mock_agent_fixture():
    """
    Fixture to create a mock agent.
    """
    return MockAgent()

@pytest.fixture
def text_search_metadata_action(mock_agent_fixture):
    """
    Fixture for setting up TextSearchMetadata action.
    """
    text_items = pd.Series(["item1", "item2", "item3"])
    metadatas = pd.Series([{"meta1": "data1"}, {"meta2": "data2"}, {"meta3": "data3"}])
    return TextSearchMetadata(mock_agent_fixture, text_items, metadatas)

@pytest.fixture
def classify_query_action(mock_agent_fixture):
    """
    Fixture for setting up ClassifyQuery action.
    """
    trigger_queries = pd.Series(["query1", "query2", "query3"])
    class_names = pd.Series(["class1", "class2", "class3"])
    return ClassifyQuery(mock_agent_fixture, trigger_queries, class_names)

@pytest.fixture
def rag_generation_action(mock_agent_fixture):
    """
    Fixture for setting up RAGGeneration action.
    """
    return RAGGeneration(mock_agent_fixture, sys_prompt="System prompt", prompt="Prompt", vector_stores=[])

@pytest.fixture
def generic_llm_response_action(mock_agent_fixture):
    """
    Fixture for setting up GenericLLMResponse action.
    """
    return GenericLLMResponse(mock_agent_fixture, sys_prompt="System prompt", prompt="Prompt")




def test_classify_query_execution(classify_query_action):
    """
    Test execution of ClassifyQuery action.
    """
    class_name, trigger_query = classify_query_action.execute(query="query1")
    
    assert class_name == "class1"
    assert trigger_query == "query1"


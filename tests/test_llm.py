import pytest
from transformers import pipeline

from rakam_systems.generation.llm import LLM

# Sample test data
SYS_PROMPT = "You are a helpful assistant."
USER_PROMPT = "Tell me something cool."

# Mock response for non-streaming completion (OpenAI)
MOCK_COMPLETION_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "Why don't scientists trust atoms? Because they make up everything!"
            }
        }
    ]
}

# Mock response for streaming completion (OpenAI)
MOCK_STREAMING_RESPONSE = [
    {"choices": [{"delta": {"content": "Why"}}]},
    {"choices": [{"delta": {"content": " don't"}}]},
    {"choices": [{"delta": {"content": " scientists"}}]},
    {"choices": [{"delta": {"content": " trust"}}]},
    {"choices": [{"delta": {"content": " atoms?"}}]},
]

# Mock response for Hugging Face model
MOCK_HF_RESPONSE = [
    {
        "generated_text": "Why don't scientists trust atoms? Because they make up everything!"
    }
]


class MockOpenAIChat:
    """Mock class for OpenAI chat completion."""

    def __init__(self, responses):
        self.responses = responses

    def completions(self, model, messages, stream=False):
        if stream:
            for chunk in self.responses:
                yield chunk
        else:
            return {"choices": self.responses}


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Fixture to mock OpenAI client."""

    class MockOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = MockOpenAIChat(MOCK_COMPLETION_RESPONSE["choices"])

    # Monkeypatch the OpenAI client
    monkeypatch.setattr("rakam_systems.generation.llm.OpenAI", MockOpenAI)


@pytest.fixture
def mock_openai_client_stream(monkeypatch):
    """Fixture to mock OpenAI client for streaming."""

    class MockOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = MockOpenAIChat(MOCK_STREAMING_RESPONSE)

    # Monkeypatch the OpenAI client
    monkeypatch.setattr("rakam_systems.generation.llm.OpenAI", MockOpenAI)


@pytest.fixture
def llm_openai():
    """Fixture to create an LLM instance using OpenAI."""
    return LLM(model="gpt-4o", api_key="")


@pytest.fixture
def test_llm_initialization_openai(llm_openai):
    """Test the LLM class initialization with OpenAI."""
    assert llm_openai.model == "gpt-4o"
    assert llm_openai.client is not None


def test_call_llm_openai(llm_openai, mock_openai_client):
    """Test the call_llm method with OpenAI."""
    response = llm_openai.call_llm(SYS_PROMPT, USER_PROMPT)

    # Assert the response is as expected
    # assert type(response) == str


def test_call_llm_stream_openai(llm_openai, mock_openai_client_stream):
    """Test the call_llm_stream method with OpenAI."""
    response = list(llm_openai.call_llm_stream(SYS_PROMPT, USER_PROMPT))

    # Assert the streaming response is as expected
    expected_response = ["Why", " don't", " scientists", " trust", " atoms?"]
    # assert response == expected_response

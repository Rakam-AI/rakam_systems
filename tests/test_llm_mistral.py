import pytest
from unittest.mock import MagicMock, patch
from rakam_systems.components.base import LLM

@pytest.fixture
def llm_mistral():
    """Fixture to create an LLM instance for Mistral."""
    return LLM(model="mistral-large-latest", api_source="Mistral")

@patch("rakam_systems.components.base.Mistral")
def test_call_llm(mock_mistral_class, llm_mistral):
    """Demonstrate the call_llm method."""
    # Mock the Mistral client
    mock_mistral_instance = MagicMock()
    mock_mistral_instance.chat.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Paris"))]
    )
    mock_mistral_class.return_value = mock_mistral_instance

    # Test the call_llm method
    response = llm_mistral.call_llm(
        sys_prompt="You are a helpful assistant", prompt="What is the capital of France?"
    )

    # Print the result
    print("Response from call_llm:", response)

@patch("rakam_systems.components.base.Mistral")
def test_call_llm_stream(mock_mistral_class, llm_mistral):
    """Demonstrate the call_llm_stream method."""
    # Mock the Mistral client
    mock_stream_response = [
        MagicMock(data=MagicMock(choices=[MagicMock(delta=MagicMock(content="Paris"))])),
        MagicMock(data=MagicMock(choices=[MagicMock(delta=MagicMock(content=" is"))])),
        MagicMock(data=MagicMock(choices=[MagicMock(delta=MagicMock(content=" beautiful"))])),
    ]
    mock_mistral_instance = MagicMock()
    mock_mistral_instance.chat.stream.return_value = mock_stream_response
    mock_mistral_class.return_value = mock_mistral_instance

    # Test the call_llm_stream method
    print("Streaming response from call_llm_stream:")
    for chunk in llm_mistral.call_llm_stream(
        sys_prompt="You are a helpful assistant", prompt="What is the capital of France?"
    ):
        print(chunk)

@patch("rakam_systems.components.base.Mistral")
def test_call_llm_output_json(mock_mistral_class, llm_mistral):
    """Demonstrate the call_llm_output_json method."""
    # This method raises an error for Mistral
    print("Calling call_llm_output_json (expected to raise ValueError):")
    try:
        llm_mistral.call_llm_output_json(
            sys_prompt="You are a helpful assistant", prompt="What is the capital of France?"
        )
    except ValueError as e:
        print("Error:", e)

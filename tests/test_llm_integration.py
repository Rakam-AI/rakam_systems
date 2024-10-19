import os
import pytest
from rakam_systems.components.base import LLM
import logging
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def llm():
    # Assuming OPENAI_API_KEY is set in your environment
    return LLM(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))

def test_call_llm(llm):
    sys_prompt = "You are an assistant."
    user_prompt = "Tell me a fun fact about pandas."

    # Make a real API call
    response = llm.call_llm(sys_prompt, user_prompt)

    # Check that the response is a non-empty string
    assert isinstance(response, str)
    assert len(response) > 0
    logging.debug(f"Response: {response}")
 
def test_call_llm_stream(llm):
    sys_prompt = "You are an assistant."
    user_prompt = "Give me three reasons why pandas are interesting."

    # Make a real API call with streaming
    response_stream = llm.call_llm_stream(sys_prompt, user_prompt)
    logging.debug(f"Stream response: {response_stream}")

    response_chunks = [chunk for chunk in response_stream if chunk]  # Filter out None values
    full_response = "".join(response_chunks)

    # Check that the response is a non-empty string
    assert isinstance(full_response, str)
    assert len(full_response) > 0
    logging.debug(f"Streamed response: {full_response}")


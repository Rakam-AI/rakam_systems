import pytest
from unittest.mock import AsyncMock, MagicMock

from rakam_systems_agent.components.base_agent import BaseAgent
from rakam_systems_core.interfaces.agent import AgentInput, AgentOutput, ModelSettings

pytestmark = pytest.mark.asyncio  # Marks all async tests in this module



@pytest.fixture
def mock_pydantic_agent(monkeypatch):
    """Patch PydanticAgent inside BaseAgent to a mock."""
    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock()
    mock_agent.run_stream = AsyncMock()
    monkeypatch.setattr(
        "rakam_systems_agent.components.base_agent.PydanticAgent", lambda **kwargs: mock_agent)
    return mock_agent


@pytest.fixture
def agent_instance(mock_pydantic_agent):
    """Return BaseAgent instance with mocks."""
    return BaseAgent(name="test_agent")



def test_init(agent_instance):
    """BaseAgent initializes properly with defaults."""
    assert agent_instance.name == "test_agent"
    assert agent_instance._pydantic_agent is not None
    assert isinstance(agent_instance._dynamic_system_prompts, list)


def test_dynamic_system_prompt_decorator(agent_instance):
    """Register dynamic system prompt with decorator."""
    calls = []

    @agent_instance.dynamic_system_prompt
    def sample_prompt(ctx=None):
        calls.append("called")
        return "hello"

    # Decorator should register function
    assert sample_prompt in agent_instance._dynamic_system_prompts


def test_dynamic_system_prompt_method(agent_instance):
    """Register dynamic system prompt via add_dynamic_system_prompt."""
    def sample_func(ctx=None):
        return "ok"

    returned = agent_instance.add_dynamic_system_prompt(sample_func)
    assert returned == sample_func
    assert sample_func in agent_instance._dynamic_system_prompts


def test_normalize_input(agent_instance):
    """_normalize_input returns AgentInput for string or passes AgentInput through."""
    ai = agent_instance._normalize_input("hello")
    assert isinstance(ai, AgentInput)
    assert ai.input_text == "hello"

    input_obj = AgentInput("world")
    ai2 = agent_instance._normalize_input(input_obj)
    assert ai2 is input_obj



async def test_arun_calls_ainfer(agent_instance, mock_pydantic_agent):
    """arun normalizes input and calls ainfer."""
    input_text = "test input"
    agent_instance.ainfer = AsyncMock(return_value=AgentOutput("ok"))
    output = await agent_instance.arun(input_text)
    agent_instance.ainfer.assert_awaited_once()
    assert output.output_text == "ok"


async def test_ainfer_returns_agent_output(agent_instance, mock_pydantic_agent):
    """ainfer converts PydanticAgent output to AgentOutput."""
    # Correctly return a MagicMock when awaited
    mock_result = MagicMock()
    mock_result.output = "response"
    mock_result.usage.return_value = {"tokens": 10}
    mock_result.all_messages.return_value = []
    mock_pydantic_agent.run = AsyncMock(return_value=mock_result)

    input_obj = AgentInput("input text")
    result = await agent_instance.ainfer(input_obj)

    assert isinstance(result, AgentOutput)
    assert result.output_text == "response"
    assert result.metadata["usage"]["tokens"] == 10


def test_infer_raises_not_implemented(agent_instance):
    """Synchronous infer should raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        agent_instance.infer(AgentInput("x"))


def test_run_raises_not_implemented(agent_instance):
    """Synchronous run should raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        agent_instance.run("x")




async def test_ainfer_forwards_message_history(agent_instance, mock_pydantic_agent):
    """ainfer passes message_history kwarg through to pydantic_agent.run."""
    mock_result = MagicMock()
    mock_result.output = "ok"
    mock_result.usage.return_value = {}
    mock_result.all_messages.return_value = []
    mock_pydantic_agent.run = AsyncMock(return_value=mock_result)

    fake_history = [MagicMock()]
    await agent_instance.ainfer(AgentInput("hi"), message_history=fake_history)

    _, kwargs = mock_pydantic_agent.run.call_args
    assert kwargs["message_history"] is fake_history


async def test_arun_forwards_message_history(agent_instance, mock_pydantic_agent):
    """arun forwards message_history through to pydantic_agent.run."""
    mock_result = MagicMock()
    mock_result.output = "ok"
    mock_result.usage.return_value = {}
    mock_result.all_messages.return_value = []
    mock_pydantic_agent.run = AsyncMock(return_value=mock_result)

    fake_history = [MagicMock()]
    await agent_instance.arun("hi", message_history=fake_history)

    _, kwargs = mock_pydantic_agent.run.call_args
    assert kwargs["message_history"] is fake_history


async def test_arun_defaults_message_history_none(agent_instance, mock_pydantic_agent):
    """arun passes message_history=None to pydantic_agent.run when not provided."""
    mock_result = MagicMock()
    mock_result.output = "ok"
    mock_result.usage.return_value = {}
    mock_result.all_messages.return_value = []
    mock_pydantic_agent.run = AsyncMock(return_value=mock_result)

    await agent_instance.arun("hi")

    _, kwargs = mock_pydantic_agent.run.call_args
    assert kwargs["message_history"] is None


async def test_astream_forwards_message_history(agent_instance, mock_pydantic_agent):
    """astream passes message_history to pydantic_agent.run_stream."""
    mock_stream_result = MagicMock()

    async def empty_chunks():
        return
        yield  # make it an async generator

    mock_stream_result.stream = empty_chunks

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_stream_result)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pydantic_agent.run_stream = MagicMock(return_value=mock_ctx)

    fake_history = [MagicMock()]
    async for _ in agent_instance.astream("hi", message_history=fake_history):
        pass

    _, kwargs = mock_pydantic_agent.run_stream.call_args
    assert kwargs["message_history"] is fake_history

import pytest
from rakam_systems_core.interfaces.agent import (AgentComponent, AgentInput,
                                                 AgentOutput, ModelSettings)


class DummyAgent(AgentComponent):
    def run(self, input_data, deps=None, model_settings=None):
        if isinstance(input_data, AgentInput):
            text = input_data.input_text
        else:
            text = input_data

        return AgentOutput(
            output_text=f"echo:{text}",
            metadata={"used_model": self.model},
            output={"structured": text.upper()},
        )

    async def arun(self, input_data, deps=None, model_settings=None):
        if isinstance(input_data, AgentInput):
            text = input_data.input_text
        else:
            text = input_data

        return AgentOutput(
            output_text=f"async:{text}",
            metadata={"async": True},
            output=text[::-1],
        )


def test_agent_input_defaults():
    inp = AgentInput("hello")

    assert inp.input_text == "hello"
    assert inp.context == {}


def test_agent_output_basic():
    out = AgentOutput("hi")

    assert out.output_text == "hi"
    assert out.metadata == {}
    assert out.output is None


def test_agent_output_structured():
    out = AgentOutput("text", output={"x": 1})

    assert out.output_text == "text"
    assert out.output == {"x": 1}


def test_model_settings_defaults():
    settings = ModelSettings()

    assert settings.parallel_tool_calls is True
    assert settings.temperature is None
    assert settings.max_tokens is None
    assert settings.extra_settings == {}


def test_model_settings_extra_kwargs():
    settings = ModelSettings(foo=123)

    assert settings.extra_settings["foo"] == 123


def test_agent_initialization_defaults():
    agent = DummyAgent("agent")

    assert agent.name == "agent"
    assert agent.model == "openai:gpt-4"
    assert agent.stateful is False
    assert agent.system_prompt == ""
    assert agent.tools == []


def test_agent_initialization_from_config():
    agent = DummyAgent(
        "agent",
        config={
            "stateful": True,
            "model": "custom-model",
            "system_prompt": "You are helpful",
        },
    )

    assert agent.stateful is True
    assert agent.model == "custom-model"
    assert agent.system_prompt == "You are helpful"


def test_run_with_string_input():
    agent = DummyAgent("agent")

    result = agent.run("hello")

    assert isinstance(result, AgentOutput)
    assert result.output_text == "echo:hello"
    assert result.metadata["used_model"] == agent.model
    assert result.output == {"structured": "HELLO"}


def test_run_with_agent_input():
    agent = DummyAgent("agent")

    inp = AgentInput("world", context={"foo": "bar"})
    result = agent.run(inp)

    assert result.output_text == "echo:world"


def test_stream_default_behavior():
    agent = DummyAgent("agent")

    chunks = list(agent.stream("hello"))

    assert chunks == ["echo:hello"]


def test_agent_call_auto_setup():
    agent = DummyAgent("agent")

    assert agent.initialized is False

    result = agent("hello")

    assert agent.initialized is True
    assert result.output_text == "echo:hello"


def test_agent_context_manager():
    agent = DummyAgent("agent")

    with agent as a:
        assert a.initialized is True

    assert agent.initialized is False


# ---------------------------------------
# Abstract Enforcement Test
# ---------------------------------------

def test_cannot_instantiate_abstract_agent():
    with pytest.raises(TypeError):
        AgentComponent("abstract")

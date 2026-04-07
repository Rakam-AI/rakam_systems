"""Tests for config_schema.py Pydantic schemas."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from rakam_systems_core.config_schema import (
    ConfigFileSchema,
    EvaluationCriteriaSchema,
    EvaluationResultSchema,
    MethodCallRecordSchema,
    MethodInputSchema,
    MethodOutputSchema,
    ModelConfigSchema,
    OutputFieldSchema,
    OutputTypeSchema,
    PromptConfigSchema,
    ToolConfigSchema,
    ToolMode,
    TrackingSessionSchema,
)


# ---------------------------------------------------------------------------
# ToolMode
# ---------------------------------------------------------------------------


def test_tool_mode_values():
    assert ToolMode.DIRECT == "direct"
    assert ToolMode.MCP == "mcp"


# ---------------------------------------------------------------------------
# ToolConfigSchema
# ---------------------------------------------------------------------------


class TestToolConfigSchema:
    def test_direct_tool_creation(self):
        tool = ToolConfigSchema(
            name="my_tool",
            type=ToolMode.DIRECT,
            description="A test tool",
            module="os.path",
            function="join",
        )
        assert tool.name == "my_tool"
        assert tool.type == "direct"
        assert tool.module == "os.path"
        assert tool.function == "join"

    def test_mcp_tool_creation(self):
        tool = ToolConfigSchema(
            name="mcp_tool",
            type=ToolMode.MCP,
            description="An MCP tool",
            mcp_server="my_server",
        )
        assert tool.name == "mcp_tool"
        assert tool.mcp_server == "my_server"

    def test_direct_tool_module_is_optional_field(self):
        # module is Optional[str] — creation without module succeeds at Pydantic level
        # (V1-style validator does not raise in Pydantic V2 for None values)
        tool = ToolConfigSchema(
            name="no_module",
            type=ToolMode.DIRECT,
            description="d",
            function="join",
        )
        assert tool.module is None

    def test_mcp_tool_server_is_optional_field(self):
        # mcp_server is Optional[str] — creation without it succeeds at Pydantic level
        tool = ToolConfigSchema(
            name="no_server",
            type=ToolMode.MCP,
            description="d",
        )
        assert tool.mcp_server is None

    def test_defaults(self):
        tool = ToolConfigSchema(
            name="t",
            type=ToolMode.DIRECT,
            description="d",
            module="os",
            function="getcwd",
        )
        assert tool.category == "general"
        assert tool.tags == []
        assert tool.takes_ctx is False

    def test_custom_tags_and_category(self):
        tool = ToolConfigSchema(
            name="t",
            type=ToolMode.DIRECT,
            description="d",
            module="os",
            function="getcwd",
            category="search",
            tags=["web", "api"],
        )
        assert tool.category == "search"
        assert tool.tags == ["web", "api"]

    def test_mcp_tool_name_defaults_to_none(self):
        tool = ToolConfigSchema(
            name="mcp",
            type=ToolMode.MCP,
            description="d",
            mcp_server="server",
        )
        assert tool.mcp_tool_name is None


# ---------------------------------------------------------------------------
# ModelConfigSchema
# ---------------------------------------------------------------------------


class TestModelConfigSchema:
    def test_basic_creation(self):
        cfg = ModelConfigSchema(model="openai:gpt-4o")
        assert cfg.model == "openai:gpt-4o"
        assert cfg.temperature is None
        assert cfg.max_tokens is None
        assert cfg.parallel_tool_calls is True

    def test_with_all_fields(self):
        cfg = ModelConfigSchema(
            model="mistral:mistral-large-latest",
            temperature=0.5,
            max_tokens=1024,
            parallel_tool_calls=False,
        )
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 1024
        assert cfg.parallel_tool_calls is False

    def test_temperature_above_max_raises(self):
        with pytest.raises(ValidationError):
            ModelConfigSchema(model="x", temperature=3.0)

    def test_temperature_below_min_raises(self):
        with pytest.raises(ValidationError):
            ModelConfigSchema(model="x", temperature=-0.1)

    def test_max_tokens_zero_raises(self):
        with pytest.raises(ValidationError):
            ModelConfigSchema(model="x", max_tokens=0)

    def test_extra_settings_default_empty(self):
        cfg = ModelConfigSchema(model="x")
        assert cfg.extra_settings == {}


# ---------------------------------------------------------------------------
# PromptConfigSchema
# ---------------------------------------------------------------------------


class TestPromptConfigSchema:
    def test_basic_creation(self):
        prompt = PromptConfigSchema(
            name="my_prompt",
            system_prompt="You are a helpful assistant.",
        )
        assert prompt.name == "my_prompt"
        assert prompt.system_prompt == "You are a helpful assistant."
        assert prompt.tags == []
        assert prompt.skills == []
        assert prompt.examples == []

    def test_with_skills_and_examples(self):
        prompt = PromptConfigSchema(
            name="p",
            system_prompt="s",
            skills=["reasoning", "coding"],
            examples=[{"user": "hello", "assistant": "hi"}],
        )
        assert "reasoning" in prompt.skills
        assert len(prompt.examples) == 1

    def test_description_optional(self):
        prompt = PromptConfigSchema(name="p", system_prompt="s")
        assert prompt.description is None


# ---------------------------------------------------------------------------
# OutputFieldSchema
# ---------------------------------------------------------------------------


class TestOutputFieldSchema:
    def test_required_field_defaults(self):
        field = OutputFieldSchema(type="str", description="The answer")
        assert field.type == "str"
        assert field.required is True
        assert field.default is None

    def test_optional_field_with_default(self):
        field = OutputFieldSchema(
            type="int", description="count", default=0, required=False
        )
        assert field.default == 0
        assert field.required is False

    def test_default_factory(self):
        field = OutputFieldSchema(
            type="list", description="items", default_factory="list"
        )
        assert field.default_factory == "list"


# ---------------------------------------------------------------------------
# OutputTypeSchema
# ---------------------------------------------------------------------------


class TestOutputTypeSchema:
    def test_creation(self):
        schema = OutputTypeSchema(
            name="MyOutput",
            fields={
                "answer": OutputFieldSchema(type="str", description="The answer"),
                "confidence": OutputFieldSchema(
                    type="float", description="Confidence", default=0.0
                ),
            },
        )
        assert schema.name == "MyOutput"
        assert "answer" in schema.fields
        assert "confidence" in schema.fields

    def test_description_optional(self):
        schema = OutputTypeSchema(
            name="M",
            fields={"x": OutputFieldSchema(type="str", description="d")},
        )
        assert schema.description is None


# ---------------------------------------------------------------------------
# ConfigFileSchema
# ---------------------------------------------------------------------------


class TestConfigFileSchema:
    def test_defaults(self):
        cfg = ConfigFileSchema()
        assert cfg.version == "1.0"
        assert cfg.tools == {}
        assert cfg.agents == {}
        assert cfg.prompts == {}
        assert cfg.global_settings == {}

    def test_with_prompts(self):
        cfg = ConfigFileSchema(
            prompts={"p1": PromptConfigSchema(name="p1", system_prompt="sys")}
        )
        assert "p1" in cfg.prompts

    def test_with_tools(self):
        cfg = ConfigFileSchema(
            tools={
                "t1": ToolConfigSchema(
                    name="t1",
                    type=ToolMode.DIRECT,
                    description="d",
                    module="os",
                    function="getcwd",
                )
            }
        )
        assert "t1" in cfg.tools


# ---------------------------------------------------------------------------
# MethodInputSchema
# ---------------------------------------------------------------------------


class TestMethodInputSchema:
    def test_creation(self):
        inp = MethodInputSchema(
            method_name="run",
            agent_name="my_agent",
            input_text="hello",
            call_id="abc123",
        )
        assert inp.method_name == "run"
        assert inp.agent_name == "my_agent"
        assert inp.input_text == "hello"
        assert inp.call_id == "abc123"
        assert isinstance(inp.timestamp, datetime)

    def test_defaults(self):
        inp = MethodInputSchema(
            method_name="m", agent_name="a", call_id="id"
        )
        assert inp.args == []
        assert inp.kwargs == {}
        assert inp.context == {}
        assert inp.parent_call_id is None


# ---------------------------------------------------------------------------
# MethodOutputSchema
# ---------------------------------------------------------------------------


class TestMethodOutputSchema:
    def test_success(self):
        out = MethodOutputSchema(
            method_name="run",
            agent_name="my_agent",
            output_text="world",
            duration_seconds=0.5,
            success=True,
            call_id="abc123",
        )
        assert out.success is True
        assert out.duration_seconds == 0.5
        assert out.error is None

    def test_failure(self):
        out = MethodOutputSchema(
            method_name="run",
            agent_name="my_agent",
            duration_seconds=0.1,
            success=False,
            error="timeout",
            call_id="abc123",
        )
        assert out.success is False
        assert out.error == "timeout"


# ---------------------------------------------------------------------------
# TrackingSessionSchema
# ---------------------------------------------------------------------------


def _make_call_record(success: bool = True) -> MethodCallRecordSchema:
    now = datetime.now()
    inp = MethodInputSchema(
        method_name="run",
        agent_name="agent",
        call_id="id1",
    )
    out = MethodOutputSchema(
        method_name="run",
        agent_name="agent",
        duration_seconds=0.5,
        success=success,
        call_id="id1",
    )
    return MethodCallRecordSchema(
        call_id="id1",
        agent_name="agent",
        method_name="run",
        input_data=inp,
        output_data=out,
        started_at=now,
        completed_at=now,
        duration_seconds=0.5,
    )


class TestTrackingSessionSchema:
    def test_add_call_success(self):
        session = TrackingSessionSchema(session_id="s1", agent_name="agent")
        record = _make_call_record(success=True)
        session.add_call(record)
        assert session.total_calls == 1
        assert session.successful_calls == 1
        assert session.failed_calls == 0
        assert session.total_duration == 0.5

    def test_add_call_failure(self):
        session = TrackingSessionSchema(session_id="s1", agent_name="agent")
        record = _make_call_record(success=False)
        session.add_call(record)
        assert session.total_calls == 1
        assert session.successful_calls == 0
        assert session.failed_calls == 1

    def test_add_multiple_calls(self):
        session = TrackingSessionSchema(session_id="s1", agent_name="agent")
        session.add_call(_make_call_record(success=True))
        session.add_call(_make_call_record(success=False))
        session.add_call(_make_call_record(success=True))
        assert session.total_calls == 3
        assert session.successful_calls == 2
        assert session.failed_calls == 1
        assert session.total_duration == pytest.approx(1.5)

    def test_end_session(self):
        session = TrackingSessionSchema(session_id="s1", agent_name="agent")
        assert session.ended_at is None
        session.end_session()
        assert session.ended_at is not None
        assert isinstance(session.ended_at, datetime)


# ---------------------------------------------------------------------------
# EvaluationCriteriaSchema
# ---------------------------------------------------------------------------


def test_evaluation_criteria_defaults():
    criteria = EvaluationCriteriaSchema(
        name="accuracy",
        description="Measures accuracy",
    )
    assert criteria.weight == 1.0
    assert criteria.min_score == 0.0
    assert criteria.max_score == 1.0


# ---------------------------------------------------------------------------
# EvaluationResultSchema
# ---------------------------------------------------------------------------


def test_evaluation_result_creation():
    result = EvaluationResultSchema(
        call_id="c1",
        evaluator="llm",
        scores={"accuracy": 0.9},
        overall_score=0.9,
        passed=True,
    )
    assert result.call_id == "c1"
    assert result.passed is True
    assert result.overall_score == 0.9
    assert result.scores["accuracy"] == 0.9


def test_evaluation_result_failure():
    result = EvaluationResultSchema(
        call_id="c2",
        evaluator="human",
        scores={"quality": 0.4},
        overall_score=0.4,
        passed=False,
        feedback="Needs improvement",
        suggestions=["Be more concise", "Add examples"],
    )
    assert result.passed is False
    assert result.feedback == "Needs improvement"
    assert len(result.suggestions) == 2

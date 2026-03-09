import pytest
from pydantic import ValidationError

from rakam_systems_tools.evaluation.schema import (
    MetricConfigBase,
    ClientSideMetricConfig,
    OCRSimilarityConfig,
    CorrectnessConfig,
    AnswerRelevancyConfig,
    FaithfulnessConfig,
    ToxicityConfig,
    JsonCorrectnessConfig,
    FieldsPresenceConfig,
    InputItem,
    TextInputItem,
    SchemaInputItem,
    EvalConfig,
    SchemaEvalConfig,
    MetricDiff,
    TestCaseComparison,
)


def test_metric_config_base():
    m = MetricConfigBase(type="custom")
    assert m.type == "custom"
    assert m.name is None


def test_metric_config_base_with_name():
    m = MetricConfigBase(type="custom", name="my_metric")
    assert m.name == "my_metric"


def test_client_side_metric_defaults():
    m = ClientSideMetricConfig(name="accuracy", score=0.9)
    assert m.success == 1
    assert m.evaluation_cost == 0
    assert m.reason is None
    assert m.threshold == 0


def test_client_side_metric_full():
    m = ClientSideMetricConfig(
        name="f1",
        score=0.75,
        success=1,
        evaluation_cost=0.01,
        reason="Looks good",
        threshold=0.7,
    )
    assert m.score == 0.75
    assert m.reason == "Looks good"


def test_ocr_similarity_defaults():
    m = OCRSimilarityConfig()
    assert m.type == "ocr_similarity"
    assert m.threshold == 0.5


def test_correctness_defaults():
    m = CorrectnessConfig()
    assert m.type == "correctness"
    assert "actual_output" in m.params
    assert "expected_output" in m.params


def test_answer_relevancy_defaults():
    m = AnswerRelevancyConfig()
    assert m.type == "answer_relevancy"
    assert m.threshold == 0.7
    assert m.include_reason is True


def test_faithfulness_defaults():
    m = FaithfulnessConfig()
    assert m.type == "faithfulness"
    assert m.threshold == 0.7


def test_toxicity_defaults():
    m = ToxicityConfig()
    assert m.type == "toxicity"
    assert m.threshold == 0.5


def test_json_correctness_requires_schema():
    with pytest.raises(ValidationError):
        JsonCorrectnessConfig()


def test_json_correctness_with_schema():
    m = JsonCorrectnessConfig(excpected_schema={"name": "string"})
    assert m.type == "json_correctness"
    assert m.excpected_schema == {"name": "string"}


def test_fields_presence_requires_schema():
    with pytest.raises(ValidationError):
        FieldsPresenceConfig()


def test_fields_presence_with_schema():
    m = FieldsPresenceConfig(excpected_schema={"field1": "string"})
    assert m.type == "fields_presence"
    assert m.strict_mode is True


def test_input_item():
    item = InputItem(input="What is 2+2?", output="4")
    assert item.input == "What is 2+2?"
    assert item.output == "4"
    assert item.metrics == []
    assert item.id is None


def test_text_input_item():
    item = TextInputItem(
        input="question",
        output="answer",
        expected_output="correct answer",
        retrieval_context=["context1"],
    )
    assert item.expected_output == "correct answer"
    assert item.retrieval_context == ["context1"]


def test_schema_input_item():
    item = SchemaInputItem(
        input="extract data",
        output='{"name": "John"}',
        expected_output='{"name": "John"}',
    )
    assert item.expected_output == '{"name": "John"}'


def test_eval_config_defaults():
    cfg = EvalConfig(data=[TextInputItem(input="hi", output="hello")])
    assert cfg.component == "unknown"
    assert cfg.label is None
    assert cfg.metrics == []


def test_eval_config_with_metrics():
    cfg = EvalConfig(
        component="my_component",
        label="run-1",
        data=[TextInputItem(input="hi", output="hello")],
        metrics=[ToxicityConfig(), AnswerRelevancyConfig()],
    )
    assert len(cfg.metrics) == 2
    assert cfg.component == "my_component"
    assert cfg.label == "run-1"


def test_eval_config_model_dump():
    cfg = EvalConfig(
        data=[TextInputItem(input="hi", output="hello")],
        metrics=[ToxicityConfig()],
    )
    d = cfg.model_dump()
    assert "data" in d
    assert "metrics" in d
    assert len(d["metrics"]) == 1


def test_schema_eval_config():
    cfg = SchemaEvalConfig(
        component="extractor",
        data=[SchemaInputItem(input="doc", output='{"key": "value"}')],
        metrics=[FieldsPresenceConfig(excpected_schema={"key": "string"})],
    )
    assert cfg.component == "extractor"
    assert len(cfg.metrics) == 1


def test_metric_diff():
    diff = MetricDiff(
        metric="toxicity",
        score_a=0.1,
        score_b=0.2,
        delta=0.1,
        success_a=True,
        success_b=True,
        threshold_a=0.5,
        threshold_b=0.5,
        status="changed",
    )
    assert diff.metric == "toxicity"
    assert diff.delta == 0.1
    assert diff.status == "changed"


def test_metric_diff_with_none_values():
    diff = MetricDiff(
        metric="correctness",
        score_a=None,
        score_b=0.9,
        delta=None,
        success_a=None,
        success_b=True,
        threshold_a=None,
        threshold_b=0.7,
        status="added",
    )
    assert diff.score_a is None
    assert diff.status == "added"


def test_test_case_comparison():
    comp = TestCaseComparison(
        testcase_a_id=1,
        testcase_b_id=2,
        metrics=[
            MetricDiff(
                metric="toxicity",
                score_a=0.1,
                score_b=0.2,
                delta=0.1,
                success_a=True,
                success_b=True,
                threshold_a=0.5,
                threshold_b=0.5,
                status="changed",
            )
        ],
    )
    assert comp.testcase_a_id == 1
    assert len(comp.metrics) == 1

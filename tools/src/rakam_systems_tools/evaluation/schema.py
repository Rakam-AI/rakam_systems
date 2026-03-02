# Common base class for all metric configs
import sys
from typing import Any, Dict, List, Literal, Optional, Union

# Base class (you can keep this abstract)
from pydantic import BaseModel, Field

if sys.version_info < (3, 9):
    from typing_extensions import Annotated
else:
    from typing import Annotated


class MetricConfigBase(BaseModel):
    type: str
    name: Optional[str] = None


class ClientSideMetricConfig(BaseModel):
    name: str
    score: float
    success: Optional[int] = 1
    evaluation_cost: Optional[float] = 0
    reason: Optional[str] = None
    threshold: Optional[float] = 0


class OCRSimilarityConfig(MetricConfigBase):
    type: Literal["ocr_similarity"] = "ocr_similarity"
    threshold: float = 0.5


class CorrectnessConfig(MetricConfigBase):
    type: Literal["correctness"] = "correctness"
    model: str = "gpt-4.1"
    steps: List[str] = Field(
        default=[
            "Check if the OCR model extracted the important information correctly. "
            "Minor formatting differences like '$1,250.00' vs '$1250.00' are acceptable."
        ]
    )
    criteria: Optional[str] = None
    params: List[Literal["actual_output", "expected_output"]] = Field(
        default=["actual_output", "expected_output"]
    )


class AnswerRelevancyConfig(MetricConfigBase):
    type: Literal["answer_relevancy"] = "answer_relevancy"
    threshold: float = 0.7
    model: str = "gpt-4.1"
    include_reason: bool = True


class FaithfulnessConfig(MetricConfigBase):
    type: Literal["faithfulness"] = "faithfulness"
    threshold: float = 0.7
    model: str = "gpt-4.1"
    include_reason: bool = True


class ToxicityConfig(MetricConfigBase):
    type: Literal["toxicity"] = "toxicity"
    threshold: float = 0.5
    model: str = "gpt-4.1"
    include_reason: bool = True


class JsonCorrectnessConfig(MetricConfigBase):
    type: Literal["json_correctness"] = "json_correctness"
    threshold: float = 0.5
    model: str = "gpt-4.1"
    include_reason: bool = True
    excpected_schema: Dict[str, Any]


class FieldsPresenceConfig(MetricConfigBase):
    type: Literal["fields_presence"] = "fields_presence"
    excpected_schema: Dict[str, Any]
    threshold: float = 0.5
    include_reason: bool = True
    strict_mode: bool = True


MetricConfig = Annotated[
    Union[
        OCRSimilarityConfig,
        CorrectnessConfig,
        AnswerRelevancyConfig,
        FaithfulnessConfig,
        ToxicityConfig,
    ],
    Field(discriminator="type"),
]

SchemaMetricConfig = Annotated[
    Union[JsonCorrectnessConfig, FieldsPresenceConfig], Field(
        discriminator="type")
]


class InputItem(BaseModel):
    id: Optional[str] = None  # set to optional to keep backward compatibility
    input: str
    output: str
    metrics: Optional[List[ClientSideMetricConfig]] = []


class TextInputItem(InputItem):
    expected_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None


class SchemaInputItem(InputItem):
    expected_output: Optional[str] = None


class EvalConfig(BaseModel):
    __eval_config__ = "text_eval"
    component: str = "unknown"
    scope: Optional[str] = None
    reason: Optional[str] = None
    risk_level: Optional[str] = None
    label: Union[str, None] = None
    data: List[TextInputItem]
    metrics: List[MetricConfig] = Field(default_factory=list)


class SchemaEvalConfig(BaseModel):
    __eval_config__ = "schema_eval"
    component: str = "unknown"
    scope: Optional[str]
    reason: Optional[str]
    risk_level: Optional[str]
    label: Union[str, None] = None
    data: List[SchemaInputItem]
    metrics: List[SchemaMetricConfig] = Field(default_factory=list)


class MetricDiff(BaseModel):
    metric: str
    score_a: Optional[float]
    score_b: Optional[float]
    delta: Optional[float]

    success_a: Optional[bool]
    success_b: Optional[bool]

    threshold_a: Optional[float]
    threshold_b: Optional[float]

    status: str  # "unchanged" | "changed" | "added" | "removed"


class TestCaseComparison(BaseModel):
    testcase_a_id: int
    testcase_b_id: int
    metrics: List[MetricDiff]

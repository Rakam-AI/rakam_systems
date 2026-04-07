import os
import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast, overload

import requests

from .schema import (
    EvalConfig,
    MetricConfig,
    SchemaEvalConfig,
    SchemaInputItem,
    SchemaMetricConfig,
    TextInputItem,
)

HTTPMethod = Literal["GET", "POST", "PATCH", "PUT", "DELETE"]


class DeepEvalClient:
    """
    Client for interacting with the DeepEval API.
    Provides synchronous and background evaluation with optional probability-based execution.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        settings_module: Optional[Any] = None,  # optional external settings
        timeout: int = 30,
    ):
        settings_url = getattr(settings_module, "EVALFRAMEWORK_URL", None)
        settings_token = getattr(settings_module, "EVALFRAMWORK_API_KEY", None)

        raw_url = (
            base_url
            or settings_url
            or os.getenv("EVALFRAMEWORK_URL")
            or "http://localhost:8080"
        )
        self.base_url = raw_url.rstrip("/")
        self.api_token = (
            api_token or settings_token or os.getenv(
                "EVALFRAMEWORK_API_KEY", "")
        )
        self.timeout = timeout

    def _request(
        self,
        method: HTTPMethod,
        endpoint: str,
        *,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
        raise_exception: bool = False,
    ) -> Optional[Dict]:
        url = f"{self.base_url}{endpoint}"

        headers = {
            "accept": "application/json",
            "X-API-Token": self.api_token,
        }

        if json is not None:
            headers["Content-Type"] = "application/json"

        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                params=params,
                timeout=self.timeout,
            )
            if raise_exception:
                resp.raise_for_status()
        except requests.RequestException as e:
            if raise_exception:
                raise
            return {"error": str(e)}

        try:
            return cast(dict, resp.json())
        except ValueError:
            if raise_exception:
                raise
            return {
                "error": "Invalid JSON response",
                "raw": resp.text,
            }

    def _get(
        self,
        endpoint: str,
        params: Dict,
        raise_exception: bool = False,
        *args: Tuple,
        **kw: Dict,
    ) -> Optional[Dict]:
        return self._request(
            "GET", endpoint, params=params, raise_exception=raise_exception, *args, **kw
        )

    def _post(
        self,
        endpoint: str,
        payload: Dict,
        raise_exception: bool = False,
        *args: Tuple,
        **kw: Dict,
    ) -> Optional[Dict]:
        return self._request(
            "POST", endpoint, json=payload, raise_exception=raise_exception, *args, **kw
        )

    def _patch(
        self,
        endpoint: str,
        payload: Dict,
        raise_exception: bool = False,
        *args: Tuple,
        **kw: Dict,
    ) -> Optional[Dict]:
        return self._request(
            "PATCH",
            endpoint,
            json=payload,
            raise_exception=raise_exception,
            *args,
            **kw,
        )

    def _delete(
        self,
        endpoint: str,
        payload: Dict,
        raise_exception: bool = False,
        *args: Tuple,
        **kw: Dict,
    ) -> Optional[Dict]:
        return self._request(
            "DELETE",
            endpoint,
            json=payload,
            raise_exception=raise_exception,
            *args,
            **kw,
        )

    def update_evaluation_testcase_tag(
        self,
        *,
        testcase_id: int,
        tag: Optional[str],
        raise_exception: bool = False,
    ) -> Optional[Dict]:
        """
        Add, update, or remove a tag from an evaluation testcase.

        - tag="smoke" → add / update tag
        - tag=None → remove tag
        """
        if testcase_id is None:
            return self._delete(
                endpoint=f"/deepeval/tag/{tag}",
                payload={},
                raise_exception=raise_exception,
            )
        return self._patch(
            endpoint=f"/deepeval/{testcase_id}/tag",
            payload={"tag": tag},
            raise_exception=raise_exception,
        )

    def list_evaluation_testcases(
        self,
        *,
        limit: int = 10,
        offset: int = 0,
        raise_exception: bool = False,
    ) -> Optional[Dict]:
        """
        List evaluation testcases for the current API token only.
        Sorted by created_at DESC (newest first).
        Returns items + pagination metadata.
        """
        return self._get(
            endpoint="/eval-framework/deepeval/evaluation-testcases/token",
            params={
                "limit": limit,
                "offset": offset,
            },
            raise_exception=raise_exception,
        )

    def get_evaluation_testcase_by_id(
        self,
        testcase_id: int,
        *,
        raise_exception: bool = False,
    ) -> Optional[Dict]:
        """
        Fetch a single evaluation testcase by numeric ID.
        """
        return self._get(
            endpoint=f"/eval-framework/deepeval/id/{testcase_id}",
            params={},
            raise_exception=raise_exception,
        )

    def get_evaluation_testcase_by_tag(
        self,
        tag: str,
        *,
        raise_exception: bool = False,
    ) -> Optional[Dict]:
        """
        Fetch a single evaluation testcase by tag.
        """
        return self._get(
            endpoint=f"/eval-framework/deepeval/tag/{tag}",
            params={},
            raise_exception=raise_exception,
        )

    def get_evaluation_testcase(
        self,
        *,
        id: Optional[int] = None,
        tag: Optional[str] = None,
        raise_exception: bool = False,
    ) -> Optional[Dict]:
        if id is not None:
            return self.get_evaluation_testcase_by_id(
                id, raise_exception=raise_exception
            )
        if tag is not None:
            return self.get_evaluation_testcase_by_tag(
                tag, raise_exception=raise_exception
            )
        raise ValueError("Either id or tag must be provided")

    def compare_testcases(
        self,
        *,
        testcase_a_id: Optional[int] = None,
        testcase_a_tag: Optional[str] = None,
        testcase_b_id: Optional[int] = None,
        testcase_b_tag: Optional[str] = None,
        raise_exception: bool = False,
    ) -> Optional[Dict]:
        """
        Compare two evaluation testcases using IDs or tags.
        Exactly one identifier (id or tag) must be provided per testcase.
        """

        def validate(id_: Optional[int], tag: Optional[str], name: str) -> None:
            if bool(id_) == bool(tag):
                raise ValueError(
                    f"Provide exactly one of {name}_id or {name}_tag")

        validate(testcase_a_id, testcase_a_tag, "testcase_a")
        validate(testcase_b_id, testcase_b_tag, "testcase_b")

        params: Dict[str, Union[int, str]] = {}

        if testcase_a_id is not None:
            params["testcase_a_id"] = testcase_a_id
        else:
            params["testcase_a_tag"] = testcase_a_tag  # type: ignore

        if testcase_b_id is not None:
            params["testcase_b_id"] = testcase_b_id
        else:
            params["testcase_b_tag"] = testcase_b_tag  # type: ignore

        return self._get(
            endpoint="/eval-framework/deepeval/evaluation-testcases/compare",
            params=params,
            raise_exception=raise_exception,
        )

    def compare_latest_by_labels(
        self,
        *,
        label_a: str,
        label_b: str,
        raise_exception: bool = False,
    ) -> Optional[Dict]:
        """
        Compare the latest evaluation testcases for two labels.
        """
        return self._get(
            endpoint="/eval-framework/deepeval/evaluation-testcases/compare-latest",
            params={
                "label_a": label_a,
                "label_b": label_b,
            },
            raise_exception=raise_exception,
        )

    def compare_last_two_by_label(
        self,
        *,
        label: str,
        raise_exception: bool = False,
    ) -> Optional[Dict]:
        """
        Compare the last two evaluation testcases for a given label.
        """
        return self._get(
            endpoint="/eval-framework/deepeval/evaluation-testcases/compare-last-two",
            params={
                "label": label,
            },
            raise_exception=raise_exception,
        )

    @overload
    def text_eval(
        self,
        config: EvalConfig,
        *,
        raise_exception: bool = False,
    ) -> Optional[Dict]: ...

    @overload
    def text_eval(
        self,
        *,
        data: List[TextInputItem],
        metrics: List[MetricConfig],
        component: str = "unknown",
        label: Optional[str] = None,
        raise_exception: bool = False,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Optional[Dict]: ...

    def text_eval(
        self,
        config: Optional[EvalConfig] = None,
        *,
        data: Optional[List[TextInputItem]] = None,
        metrics: Optional[List[MetricConfig]] = None,
        component: str = "unknown",
        label: Optional[str] = None,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        risk_level: Optional[str] = None,
        raise_exception: bool = False,
    ) -> Optional[Dict]:
        if config is None:
            config = EvalConfig(
                data=data,
                metrics=metrics,
                component=component,
                label=label,
                scope=scope,
                reason=reason,
                risk_level=risk_level
            )

        return self._post(
            endpoint="/deepeval/text-eval",
            payload=config.model_dump(),
            raise_exception=raise_exception,
        )

    def text_eval_background(
        self,
        data: List[TextInputItem],
        metrics: List[MetricConfig],
        raise_exception: bool = False,
        component: str = "unknown",
        label: Union[str, None] = None,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Optional[Dict]:
        """Run background text evaluation (async job)."""
        payload = EvalConfig.model_construct(
            data=data,
            metrics=metrics,
            component=component,
            version=label,
            scope=scope,
            reason=reason,
            risk_level=risk_level

        ).model_dump()
        return self._post(
            endpoint="/deepeval/text-eval/background",
            payload=payload,
            raise_exception=raise_exception,
        )

    @overload
    def schema_eval(
        self,
        config: SchemaEvalConfig,
        *,
        raise_exception: bool = False,

    ) -> Optional[Dict]: ...

    @overload
    def schema_eval(
        self,
        *,
        data: List[SchemaInputItem],
        metrics: List[SchemaMetricConfig],
        component: str = "unknown",
        label: Optional[str] = None,
        raise_exception: bool = False,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Optional[Dict]: ...

    def schema_eval(
        self,
        config: Optional[SchemaEvalConfig] = None,
        *,
        data: Optional[List[SchemaInputItem]] = None,
        metrics: Optional[List[SchemaMetricConfig]] = None,
        component: str = "unknown",
        label: Optional[str] = None,
        raise_exception: bool = False,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Optional[Dict]:
        if config is None:
            if data is None or metrics is None:
                raise ValueError(
                    "Either `config` or both `data` and `metrics` must be provided"
                )

            config = SchemaEvalConfig(
                data=data,
                metrics=metrics,
                component=component,
                label=label,
                scope=scope,
                reason=reason,
                risk_level=risk_level

            )

        return self._post(
            endpoint="/deepeval/schema-eval",
            payload=config.model_dump(),
            raise_exception=raise_exception,
        )

    def schema_eval_background(
        self,
        data: List[SchemaInputItem],
        metrics: List[SchemaMetricConfig],
        raise_exception: bool = False,
        component: str = "unknown",
        label: Union[str, None] = None,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Optional[Dict]:
        """Run background schema evaluation (async job)."""
        payload = SchemaEvalConfig.model_construct(
            data=data, metrics=metrics, component=component, version=label,
            scope=scope,
            reason=reason,
            risk_level=risk_level
        ).model_dump()
        return self._post(
            endpoint="/deepeval/schema-eval/background",
            payload=payload,
            raise_exception=raise_exception,
        )

    def maybe_text_eval(
        self,
        data: List[TextInputItem],
        metrics: List[MetricConfig],
        chance: float,
        raise_exception: bool = False,
        component: str = "unknown",
        label: Union[str, None] = None,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Optional[Dict]:
        """Randomly run text_eval based on a probability between 0 and 1."""
        self._validate_chance(chance)
        return (
            self.text_eval(
                data=data,
                metrics=metrics,
                raise_exception=raise_exception,
                component=component,
                label=label,
                scope=scope,
                reason=reason,
                risk_level=risk_level
            )
            if random.random() <= chance
            else None
        )

    def maybe_text_eval_background(
        self,
        data: List[TextInputItem],
        metrics: List[MetricConfig],
        chance: float,
        raise_exception: bool = False,
        component: str = "unknown",
        label: Union[str, None] = None,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Optional[Dict]:
        """Randomly run text_eval_background based on a probability between 0 and 1."""
        self._validate_chance(chance)
        return (
            self.text_eval_background(
                data, metrics, raise_exception, component=component, label=label,
                scope=scope,
                reason=reason,
                risk_level=risk_level
            )
            if random.random() <= chance
            else None
        )

    def maybe_schema_eval(
        self,
        data: List[SchemaInputItem],
        metrics: List[SchemaMetricConfig],
        chance: float,
        raise_exception: bool = False,
        component: str = "unknown",
        label: Union[str, None] = None,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Optional[Dict]:
        """Randomly run schema_eval based on a probability between 0 and 1."""
        self._validate_chance(chance)
        return (
            self.schema_eval(
                data=data,
                metrics=metrics,
                raise_exception=raise_exception,
                component=component,
                label=label,
                scope=scope,
                reason=reason,
                risk_level=risk_level
            )
            if random.random() <= chance
            else None
        )

    def maybe_schema_eval_background(
        self,
        data: List[SchemaInputItem],
        metrics: List[SchemaMetricConfig],
        chance: float,
        raise_exception: bool = False,
        component: str = "unknown",
        label: Union[str, None] = None,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Optional[Dict]:
        """Randomly run text_eval_background based on a probability between 0 and 1."""
        self._validate_chance(chance)
        return (
            self.schema_eval_background(
                data, metrics, raise_exception, component=component, label=label,
                scope=scope,
                reason=reason,
                risk_level=risk_level
            )
            if random.random() <= chance
            else None
        )

    @staticmethod
    def _validate_chance(chance: float) -> None:
        """Ensure chance is a valid probability between 0 and 1."""
        if not (0 <= chance <= 1):
            raise ValueError("chance must be between 0 and 1.")

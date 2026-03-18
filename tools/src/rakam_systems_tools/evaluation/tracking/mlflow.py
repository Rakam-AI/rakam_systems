import os
from typing import Any, Dict, List, Literal, Optional

from .base import EvaluationTracker


class MLflowTracker(EvaluationTracker):
    """Evaluation tracker backed by the MLflow SDK."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "mlflow is required. Install it with: pip install 'rakam-systems-tools[mlflow]'"
            ) from e

        self._mlflow = mlflow
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)

        self._experiment_id = experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")

    def log_trace(
        self,
        name: str,
        input: Dict[str, Any],
        output: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        tag_dict = {t: "" for t in tags} if tags else {}
        if metadata:
            tag_dict.update(metadata)

        with self._mlflow.start_trace(name=name) as trace:
            with self._mlflow.start_span(name="llm_call") as span:
                span.set_inputs(input)
                span.set_outputs(output)
                if tag_dict:
                    for k, v in tag_dict.items():
                        self._mlflow.set_tag(k, v)
            return trace.info.request_id

    def fetch_traces(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        experiment_ids = [self._experiment_id] if self._experiment_id else None
        filter_parts = []
        if name:
            filter_parts.append(f"name = '{name}'")
        filter_string = " AND ".join(filter_parts) if filter_parts else None

        traces = self._mlflow.search_traces(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=limit,
        )
        return traces.to_dict(orient="records") if hasattr(traces, "to_dict") else list(traces)

    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        trace = self._mlflow.get_trace(trace_id)
        return trace.__dict__ if hasattr(trace, "__dict__") else trace

    def log_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        source_type: Literal["HUMAN", "LLM_JUDGE", "CODE"] = "CODE",
    ) -> None:
        self._mlflow.log_feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            rationale=comment,
            source=source_type.lower(),
        )

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError(
            "MLflow does not support named datasets in the same way as Langfuse. "
            "Use MLflow's experiment tracking or artifact logging instead."
        )

    def add_dataset_item(
        self,
        dataset_name: str,
        input: Dict[str, Any],
        expected_output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        raise NotImplementedError(
            "MLflow does not support named datasets. Use create_dataset() is not supported on this backend."
        )

    def get_dataset(self, name: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "MLflow does not support named datasets. Use create_dataset() is not supported on this backend."
        )

    def list_datasets(self) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "MLflow does not support named datasets. Use create_dataset() is not supported on this backend."
        )

    def evaluate_traces(
        self,
        trace_ids: List[str],
        scorers: List[Any],
    ) -> List[Dict[str, Any]]:
        results = self._mlflow.genai.evaluate(data=trace_ids, scorers=scorers)
        if hasattr(results, "to_dict"):
            return results.to_dict(orient="records")
        return list(results)

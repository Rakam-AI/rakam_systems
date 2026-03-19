import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Literal, Optional

from .base import EvaluationTracker, SpanHandle, TraceHandle


# Map unified span_type strings to MLflow SpanType constants.
_MLF_SPAN_TYPE: Dict[str, str] = {
    "span": "UNKNOWN",
    "chain": "CHAIN",
    "retriever": "RETRIEVER",
    "generation": "LLM",
    "tool": "TOOL",
    "agent": "AGENT",
    "embedding": "EMBEDDING",
    "evaluator": "EVALUATOR",
    "guardrail": "GUARDRAIL",
}


class _MLflowSpanHandle(SpanHandle):
    def __init__(self, mlflow: Any, span: Any) -> None:
        self._mlflow = mlflow
        self._span = span

    def set_output(
        self,
        output: Dict[str, Any],
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._span.set_outputs(output)
        if usage:
            inp = usage.get("input_tokens")
            out = usage.get("output_tokens")
            total = usage.get("total_tokens") or ((inp or 0) + (out or 0))
            token_usage: Dict[str, int] = {
                k: int(v)
                for k, v in {"input_tokens": inp, "output_tokens": out, "total_tokens": total}.items()
                if v is not None
            }
            if token_usage:
                self._span.set_attribute("mlflow.chat.tokenUsage", token_usage)

            cost_inp = usage.get("input_cost")
            cost_out = usage.get("output_cost")
            cost_total = usage.get("total_cost") or (
                (cost_inp or 0.0) + (cost_out or 0.0)
                if (cost_inp is not None or cost_out is not None) else None
            )
            cost: Dict[str, float] = {
                k: float(v)
                for k, v in {"input_cost": cost_inp, "output_cost": cost_out, "total_cost": cost_total}.items()
                if v is not None
            }
            if cost:
                self._span.set_attribute("mlflow.llm.cost", cost)

            if usage.get("model"):
                self._span.set_attribute("mlflow.llm.model", usage["model"])


class _MLflowTraceHandle(TraceHandle):
    """Uses the root LiveSpan directly — avoids the removed mlflow.start_trace() API."""

    def __init__(self, mlflow: Any, root_span: Any, assessment_source: Any) -> None:
        self._mlflow = mlflow
        self._root_span = root_span  # LiveSpan — the root CHAIN span
        self._AssessmentSource = assessment_source

    @property
    def trace_id(self) -> str:
        return self._root_span.trace_id

    def set_output(self, output: Dict[str, Any]) -> None:
        self._root_span.set_outputs(output)

    def add_score(
        self,
        name: str,
        value: float,
        comment: Optional[str] = None,
        source_type: Literal["HUMAN", "LLM_JUDGE", "CODE"] = "CODE",
    ) -> None:
        self._mlflow.log_feedback(
            trace_id=self._root_span.trace_id,
            name=name,
            value=value,
            rationale=comment,
            source=self._AssessmentSource(source_type=source_type),
        )

    @contextmanager
    def span(
        self,
        name: str,
        input: Dict[str, Any],
        span_type: str = "span",
    ) -> Iterator[_MLflowSpanHandle]:
        mlf_type = _MLF_SPAN_TYPE.get(span_type, "UNKNOWN")
        with self._mlflow.start_span(name=name, span_type=mlf_type) as s:
            s.set_inputs(input)
            yield _MLflowSpanHandle(self._mlflow, s)


class MLflowTracker(EvaluationTracker):
    """Evaluation tracker backed by the MLflow SDK.

    Sessions are emulated via a 'session_id' tag on each trace (MLflow has no
    native session concept). get_session() and list_sessions() search by that tag.
    """

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

        from mlflow.entities.assessment_source import AssessmentSource
        from mlflow.entities.trace_location import MlflowExperimentLocation

        self._mlflow = mlflow
        self._AssessmentSource = AssessmentSource
        self._MlflowExperiment = MlflowExperimentLocation
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)

        self._experiment_id = experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")

    def _trace_destination(self) -> Optional[Any]:
        """Return a MlflowExperimentLocation destination when experiment_id is configured."""
        if self._experiment_id:
            return self._MlflowExperiment(experiment_id=self._experiment_id)
        return None

    def flush(self) -> None:
        """Flush all pending async trace exports to the tracking server."""
        self._mlflow.flush_trace_async_logging(terminate=False)

    @contextmanager
    def start_trace(
        self,
        name: str,
        input: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Iterator[_MLflowTraceHandle]:
        tag_dict = {t: "" for t in tags} if tags else {}
        trace_metadata: Dict[str, Any] = {}
        if session_id:
            trace_metadata["mlflow.trace.session"] = session_id
        if user_id:
            trace_metadata["mlflow.trace.user"] = user_id

        # mlflow.start_trace() was removed in MLflow 3.x.
        # mlflow.start_span() with no active parent creates a root span (= new trace).
        with self._mlflow.start_span(name=name, span_type="CHAIN",
                                     trace_destination=self._trace_destination()) as root_span:
            root_span.set_inputs(input)
            yield _MLflowTraceHandle(self._mlflow, root_span, self._AssessmentSource)
            self._mlflow.update_current_trace(
                tags=tag_dict if tag_dict else None,
                metadata=trace_metadata if trace_metadata else None,
            )

    def log_trace(
        self,
        name: str,
        input: Dict[str, Any],
        output: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> str:
        tag_dict = {t: "" for t in tags} if tags else {}

        # Build trace metadata — mlflow.trace.session and mlflow.trace.user are
        # the native MLflow keys that power the Sessions / Users views in the UI.
        trace_metadata: Dict[str, Any] = dict(metadata) if metadata else {}
        if session_id:
            trace_metadata["mlflow.trace.session"] = session_id
        if user_id:
            trace_metadata["mlflow.trace.user"] = user_id

        # mlflow.start_span() with no active parent creates a root span (= new trace).
        with self._mlflow.start_span(name=name, span_type="CHAIN",
                                     trace_destination=self._trace_destination()) as root_span:
            with self._mlflow.start_span(name="llm_call", span_type="LLM") as span:
                span.set_inputs(input)
                span.set_outputs(output)
                if usage:
                    # mlflow.chat.tokenUsage — powers the token-count display in the UI
                    inp = usage.get("input_tokens")
                    out = usage.get("output_tokens")
                    total = usage.get("total_tokens") or (
                        (inp or 0) + (out or 0) if (inp is not None or out is not None) else None
                    )
                    token_usage: Dict[str, int] = {
                        k: int(v)
                        for k, v in {"input_tokens": inp, "output_tokens": out, "total_tokens": total}.items()
                        if v is not None
                    }
                    if token_usage:
                        span.set_attribute("mlflow.chat.tokenUsage", token_usage)

                    # mlflow.llm.cost — powers the cost display in the UI
                    cost_inp = usage.get("input_cost")
                    cost_out = usage.get("output_cost")
                    total_cost = usage.get("total_cost") or (
                        (cost_inp or 0.0) + (cost_out or 0.0)
                        if (cost_inp is not None or cost_out is not None) else None
                    )
                    cost: Dict[str, float] = {
                        k: float(v)
                        for k, v in {"input_cost": cost_inp, "output_cost": cost_out, "total_cost": total_cost}.items()
                        if v is not None
                    }
                    if cost:
                        span.set_attribute("mlflow.llm.cost", cost)

                    if usage.get("model"):
                        span.set_attribute("mlflow.llm.model", usage["model"])

            self._mlflow.update_current_trace(
                tags=tag_dict if tag_dict else None,
                metadata=trace_metadata if trace_metadata else None,
            )
            return root_span.trace_id

    def fetch_traces(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        locations = [self._experiment_id] if self._experiment_id else None
        filter_parts = []
        if name:
            filter_parts.append(f"name = '{name}'")
        if session_id:
            filter_parts.append(f"metadata.`mlflow.trace.session` = '{session_id}'")
        if user_id:
            filter_parts.append(f"metadata.`mlflow.trace.user` = '{user_id}'")
        filter_string = " AND ".join(filter_parts) if filter_parts else None

        traces = self._mlflow.search_traces(
            locations=locations,
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
            source=self._AssessmentSource(source_type=source_type),
        )

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Return all traces for this session (filtered by mlflow.trace.session metadata)."""
        traces = self.fetch_traces(session_id=session_id, limit=1000)
        return {"session_id": session_id, "traces": traces}

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return distinct session IDs from recent traces (via mlflow.trace.session metadata)."""
        locations = [self._experiment_id] if self._experiment_id else None
        try:
            traces = self._mlflow.search_traces(
                locations=locations,
                filter_string="metadata.`mlflow.trace.session` != ''",
                max_results=limit * 10,
            )
            if hasattr(traces, "to_dict"):
                rows = traces.to_dict(orient="records")
            else:
                rows = list(traces)

            seen: dict[str, int] = {}
            for row in rows:
                meta = row.get("metadata", {}) if isinstance(row, dict) else {}
                sid = meta.get("mlflow.trace.session") if isinstance(meta, dict) else None
                if sid:
                    seen[sid] = seen.get(sid, 0) + 1

            return [{"session_id": sid, "trace_count": cnt} for sid, cnt in list(seen.items())[:limit]]
        except Exception:
            return []

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

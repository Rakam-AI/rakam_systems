import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Literal, Optional

from .base import EvaluationTracker, SpanHandle, TraceHandle


# Langfuse span_type strings accepted by start_as_current_observation's as_type param.
# generation / embedding are "generation" types; the rest are "span" subtypes.
_LF_SPAN_TYPE: Dict[str, str] = {
    "span": "span",
    "chain": "chain",
    "retriever": "retriever",
    "generation": "generation",
    "tool": "tool",
    "agent": "agent",
    "embedding": "embedding",
    "evaluator": "evaluator",
    "guardrail": "guardrail",
}


class _LangfuseSpanHandle(SpanHandle):
    """Uses the span object directly — avoids relying on OTel 'current span' helpers."""

    def __init__(self, span: Any) -> None:
        self._span = span  # LangfuseObservationWrapper

    def set_output(
        self,
        output: Dict[str, Any],
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        if usage:
            inp = usage.get("input_tokens")
            out = usage.get("output_tokens")
            total = usage.get("total_tokens") or ((inp or 0) + (out or 0))
            cost_inp = usage.get("input_cost")
            cost_out = usage.get("output_cost")
            cost_total = usage.get("total_cost") or (
                (cost_inp or 0.0) + (cost_out or 0.0)
                if (cost_inp is not None or cost_out is not None) else None
            )
            usage_details = {k: v for k, v in {"input": inp, "output": out, "total": total}.items() if v is not None}
            cost_details = {k: v for k, v in {"input": cost_inp, "output": cost_out, "total": cost_total}.items() if v is not None} or None
            self._span.update(
                output=output,
                model=usage.get("model"),
                usage_details=usage_details or None,
                cost_details=cost_details or None,
            )
        else:
            self._span.update(output=output)


class _LangfuseTraceHandle(TraceHandle):
    """Uses the root span object directly — avoids relying on OTel 'current span' helpers."""

    def __init__(self, lf_client: Any, root_span: Any) -> None:
        self._lf = lf_client
        self._root_span = root_span  # LangfuseObservationWrapper for the root chain

    @property
    def trace_id(self) -> Optional[str]:
        return self._root_span.trace_id

    def set_output(self, output: Dict[str, Any]) -> None:
        self._root_span.update(output=output)

    def add_score(
        self,
        name: str,
        value: float,
        comment: Optional[str] = None,
        source_type: Literal["HUMAN", "LLM_JUDGE", "CODE"] = "CODE",
    ) -> None:
        self._root_span.score_trace(name=name, value=value, comment=comment)

    @contextmanager
    def span(
        self,
        name: str,
        input: Dict[str, Any],
        span_type: str = "span",
    ) -> Iterator[_LangfuseSpanHandle]:
        as_type = _LF_SPAN_TYPE.get(span_type, "span")
        with self._lf.start_as_current_observation(name=name, as_type=as_type, input=input) as child_span:
            yield _LangfuseSpanHandle(child_span)


class LangfuseTracker(EvaluationTracker):
    """Evaluation tracker backed by the Langfuse SDK (v4+)."""

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        try:
            from langfuse import Langfuse
        except ImportError as e:
            raise ImportError(
                "langfuse is required. Install it with: pip install 'rakam-systems-tools[langfuse]'"
            ) from e

        self._client = Langfuse(
            public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY"),
            host=host or os.environ.get("LANGFUSE_HOST"),
        )

    @contextmanager
    def start_trace(
        self,
        name: str,
        input: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Iterator[_LangfuseTraceHandle]:
        from langfuse import propagate_attributes

        with propagate_attributes(session_id=session_id, user_id=user_id, tags=tags):
            with self._client.start_as_current_observation(name=name, as_type="chain", input=input) as root_span:
                yield _LangfuseTraceHandle(self._client, root_span)

    def flush(self) -> None:
        """Flush all pending trace exports to the Langfuse server."""
        self._client.flush()

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
        from langfuse import propagate_attributes

        # Build Langfuse-native token/cost dicts from the backend-agnostic usage dict.
        # as_type="generation" enables the token-count and cost UI in Langfuse.
        model: Optional[str] = None
        usage_details: Optional[Dict[str, int]] = None
        cost_details: Optional[Dict[str, float]] = None

        if usage:
            model = usage.get("model")
            inp = usage.get("input_tokens")
            out = usage.get("output_tokens")
            total = usage.get("total_tokens") or (
                (inp or 0) + (out or 0) if (inp is not None or out is not None) else None
            )
            usage_details = {k: v for k, v in {"input": inp, "output": out, "total": total}.items() if v is not None}  # type: ignore[misc]
            cost_inp = usage.get("input_cost")
            cost_out = usage.get("output_cost")
            cost_total = usage.get("total_cost") or (
                (cost_inp or 0.0) + (cost_out or 0.0) if (cost_inp is not None or cost_out is not None) else None
            )
            cost_details = {k: v for k, v in {"input": cost_inp, "output": cost_out, "total": cost_total}.items() if v is not None}  # type: ignore[misc]
            if not cost_details:
                cost_details = None

        # propagate_attributes sets trace-level context (session_id, user_id, tags).
        with propagate_attributes(session_id=session_id, user_id=user_id, tags=tags):
            span = self._client.start_observation(
                name=name,
                # "generation" enables the token-count and cost display in the Langfuse UI
                as_type="generation" if usage else "span",
                input=input,
                output=output,
                metadata=metadata,
                model=model,
                usage_details=usage_details or None,
                cost_details=cost_details,
            )
            trace_id = span.trace_id
            span.end()

        return trace_id

    def fetch_traces(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        response = self._client.api.trace.list(
            name=name,
            tags=tags,
            limit=limit,
            session_id=session_id,
            user_id=user_id,
        )
        items = getattr(response, "data", response)
        return [item.__dict__ if hasattr(item, "__dict__") else item for item in items]

    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        trace = self._client.api.trace.get(trace_id)
        return trace.__dict__ if hasattr(trace, "__dict__") else trace

    def log_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        source_type: Literal["HUMAN", "LLM_JUDGE", "CODE"] = "CODE",
    ) -> None:
        self._client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )

    def get_session(self, session_id: str) -> Dict[str, Any]:
        session = self._client.api.sessions.get(session_id)
        return session.__dict__ if hasattr(session, "__dict__") else session

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        response = self._client.api.sessions.list(limit=limit)
        items = getattr(response, "data", response)
        return [item.__dict__ if hasattr(item, "__dict__") else item for item in items]

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        dataset = self._client.create_dataset(
            name=name,
            description=description,
            metadata=metadata,
        )
        return dataset.name

    def add_dataset_item(
        self,
        dataset_name: str,
        input: Dict[str, Any],
        expected_output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        item = self._client.create_dataset_item(
            dataset_name=dataset_name,
            input=input,
            expected_output=expected_output,
            metadata=metadata,
            id=item_id,
        )
        return item.id

    def get_dataset(self, name: str) -> Dict[str, Any]:
        dataset = self._client.get_dataset(name)
        return dataset.__dict__ if hasattr(dataset, "__dict__") else dataset

    def list_datasets(self) -> List[Dict[str, Any]]:
        response = self._client.api.datasets.list()
        items = getattr(response, "data", response)
        return [item.__dict__ if hasattr(item, "__dict__") else item for item in items]

    def evaluate_traces(
        self,
        trace_ids: List[str],
        scorers: List[Any],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "evaluate_traces is not natively supported by Langfuse. "
            "Use log_score() to record individual scores per trace."
        )

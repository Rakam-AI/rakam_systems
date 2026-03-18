import os
from typing import Any, Dict, List, Literal, Optional

from .base import EvaluationTracker


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

    def log_trace(
        self,
        name: str,
        input: Dict[str, Any],
        output: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        # v4: start_observation creates a new root span (and implicitly a new trace).
        # set_trace_io promotes input/output to the trace level in the Langfuse UI.
        span = self._client.start_observation(
            name=name,
            input=input,
            output=output,
            metadata=metadata,
        )
        trace_id = span.trace_id
        span.set_trace_io(input=input, output=output)

        if tags:
            # _create_trace_tags_via_ingestion is the only way to attach string tags
            # to a trace in v4 outside of the @observe decorator.
            self._client._create_trace_tags_via_ingestion(
                trace_id=trace_id, tags=tags
            )

        span.end()
        return trace_id

    def fetch_traces(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        # api.trace.list accepts tags as a list[str] or a comma-separated string
        response = self._client.api.trace.list(name=name, tags=tags, limit=limit)
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
        # v4: create_score replaces the old score() method
        self._client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )

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

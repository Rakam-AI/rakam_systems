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
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        from langfuse import propagate_attributes

        # propagate_attributes is a context manager that sets trace-level context
        # (session_id, user_id, tags) on every observation created within the block.
        with propagate_attributes(session_id=session_id, user_id=user_id, tags=tags):
            span = self._client.start_observation(
                name=name,
                input=input,
                output=output,
                metadata=metadata,
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

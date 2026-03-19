from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Optional


class SpanHandle(ABC):
    """Handle for an active child span, yielded by ``TraceHandle.span()``."""

    @abstractmethod
    def set_output(
        self,
        output: Dict[str, Any],
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set span output.  Pass ``usage`` on generation spans for token/cost tracking.

        ``usage`` keys (all optional):
            model, input_tokens, output_tokens, total_tokens,
            input_cost, output_cost, total_cost
        """


class TraceHandle(ABC):
    """Handle for an active root trace, yielded by ``EvaluationTracker.start_trace()``."""

    @property
    @abstractmethod
    def trace_id(self) -> str:
        """ID of the root trace (available immediately after entering the context)."""

    @abstractmethod
    def set_output(self, output: Dict[str, Any]) -> None:
        """Set the root trace output (can be called at any point before exit)."""

    @abstractmethod
    def add_score(
        self,
        name: str,
        value: float,
        comment: Optional[str] = None,
        source_type: Literal["HUMAN", "LLM_JUDGE", "CODE"] = "CODE",
    ) -> None:
        """Attach a numeric score to the root trace."""

    @abstractmethod
    def span(
        self,
        name: str,
        input: Dict[str, Any],
        span_type: str = "span",
    ) -> "Iterator[SpanHandle]":
        """Context manager that creates a child span nested under this trace.

        ``span_type`` is a backend-agnostic label:
            ``"span"``       — generic step
            ``"chain"``      — multi-step chain / sub-pipeline
            ``"retriever"``  — document retrieval
            ``"generation"`` — LLM call (enables token/cost display in the UI)
            ``"tool"``       — tool / function call
            ``"agent"``      — agent sub-run

        Usage::

            with trace.span("retrieve", input={"query": q}, span_type="retriever") as span:
                docs = fetch_docs(q)
                span.set_output({"docs": docs})
        """


class EvaluationTracker(ABC):
    """Abstract base class for evaluation tracking backends."""

    def flush(self) -> None:
        """Flush any pending async trace exports. No-op by default."""

    @abstractmethod
    def start_trace(
        self,
        name: str,
        input: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "Iterator[TraceHandle]":
        """Context manager that creates a root trace with support for nested child spans.

        Yields a :class:`TraceHandle` that exposes ``.span()`` for nesting,
        ``.set_output()``, ``.add_score()``, and ``.trace_id``.

        Usage::

            with tracker.start_trace("pipeline", input={...}, session_id="s1") as trace:
                with trace.span("retrieve", input={...}, span_type="retriever") as span:
                    docs = fetch_docs(...)
                    span.set_output({"docs": docs})
                with trace.span("generate", input={...}, span_type="generation") as span:
                    answer, usage = llm_call(...)
                    span.set_output({"answer": answer}, usage=usage)
                trace.set_output({"answer": answer})
                trace.add_score("quality", 0.9)
        """

    @abstractmethod
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
        """Log a trace and return its trace_id.

        ``usage`` is a backend-agnostic token/cost dict with optional keys:
            model         str   — model name
            input_tokens  int   — prompt / input token count
            output_tokens int   — completion / output token count
            total_tokens  int   — total token count (computed if omitted)
            input_cost    float — cost of input tokens in USD
            output_cost   float — cost of output tokens in USD
            total_cost    float — total cost in USD
        """

    @abstractmethod
    def fetch_traces(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch traces filtered by name/tags/session."""

    @abstractmethod
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Fetch a single trace by ID."""

    @abstractmethod
    def log_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        source_type: Literal["HUMAN", "LLM_JUDGE", "CODE"] = "CODE",
    ) -> None:
        """Log a score/feedback for a trace."""

    @abstractmethod
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Fetch a session and its traces by session ID."""

    @abstractmethod
    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List sessions."""

    @abstractmethod
    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a dataset and return its id/name."""

    @abstractmethod
    def add_dataset_item(
        self,
        dataset_name: str,
        input: Dict[str, Any],
        expected_output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        """Add an item to a dataset and return its id."""

    @abstractmethod
    def get_dataset(self, name: str) -> Dict[str, Any]:
        """Fetch a dataset by name."""

    @abstractmethod
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets."""

    @abstractmethod
    def evaluate_traces(
        self,
        trace_ids: List[str],
        scorers: List[Any],
    ) -> List[Dict[str, Any]]:
        """Run evaluation scorers over traces and return results."""

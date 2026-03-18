from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional


class EvaluationTracker(ABC):
    """Abstract base class for evaluation tracking backends."""

    @abstractmethod
    def log_trace(
        self,
        name: str,
        input: Dict[str, Any],
        output: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Log a trace and return its trace_id."""

    @abstractmethod
    def fetch_traces(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch traces filtered by name/tags."""

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

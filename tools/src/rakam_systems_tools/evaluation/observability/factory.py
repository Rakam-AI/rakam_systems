from typing import Any, Literal

from .base import EvaluationTracker


def create_tracker(
    backend: Literal["langfuse", "mlflow"],
    **kwargs: Any,
) -> EvaluationTracker:
    """Create and return an EvaluationTracker for the given backend.

    Args:
        backend: "langfuse" or "mlflow"
        **kwargs: Passed to the backend constructor.
            LangfuseTracker accepts: public_key, secret_key, host
            MLflowTracker accepts: tracking_uri, experiment_id

    Returns:
        An EvaluationTracker instance.
    """
    if backend == "langfuse":
        from .langfuse import LangfuseTracker

        return LangfuseTracker(**kwargs)
    elif backend == "mlflow":
        from .mlflow import MLflowTracker

        return MLflowTracker(**kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'langfuse' or 'mlflow'.")

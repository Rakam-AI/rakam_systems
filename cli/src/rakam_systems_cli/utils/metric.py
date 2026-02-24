from typing import Any, List, Optional, Tuple


def extract_metric_names(config: Any) -> List[Tuple[str, Optional[str]]]:
    """
    Returns [(type, name)] from EvalConfig / SchemaEvalConfig
    """
    if not hasattr(config, "metrics"):
        return []

    results: List[Tuple[str, Optional[str]]] = []

    for metric in config.metrics or []:
        metric_type = getattr(metric, "type", None)
        metric_name = getattr(metric, "name", None)
        if metric_type:
            results.append((metric_type, metric_name))

    return results

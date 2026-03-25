from types import SimpleNamespace

from rakam_systems_cli.utils.metric import extract_metric_names


def test_extract_metric_names_no_metrics_attr() -> None:
    config = SimpleNamespace()

    result = extract_metric_names(config)

    assert result == []


def test_extract_metric_names_empty_metrics() -> None:
    config = SimpleNamespace(metrics=[])

    result = extract_metric_names(config)

    assert result == []


def test_extract_metric_names_with_types_only() -> None:
    config = SimpleNamespace(
        metrics=[
            SimpleNamespace(type="accuracy"),
            SimpleNamespace(type="latency"),
        ]
    )

    result = extract_metric_names(config)

    assert result == [
        ("accuracy", None),
        ("latency", None),
    ]


def test_extract_metric_names_with_type_and_name() -> None:
    config = SimpleNamespace(
        metrics=[
            SimpleNamespace(type="accuracy", name="acc_v1"),
            SimpleNamespace(type="f1", name=None),
        ]
    )

    result = extract_metric_names(config)

    assert result == [
        ("accuracy", "acc_v1"),
        ("f1", None),
    ]

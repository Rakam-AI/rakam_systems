import pytest

from rakam_systems_tools.utils import metrics, tracing


@pytest.fixture(autouse=True)
def clear_metrics():
    metrics._METRICS.clear()
    yield
    metrics._METRICS.clear()


def test_record_metric():
    metrics.record_metric("accuracy", 0.95)
    assert metrics._METRICS["accuracy"] == 0.95


def test_get_metric_existing():
    metrics.record_metric("latency", 42.5)
    val = metrics.get_metric("latency")
    assert val == 42.5


def test_get_metric_nonexistent():
    val = metrics.get_metric("does_not_exist")
    assert val is None


def test_record_metric_overwrites():
    metrics.record_metric("score", 0.5)
    metrics.record_metric("score", 0.8)
    assert metrics.get_metric("score") == 0.8


def test_multiple_metrics():
    metrics.record_metric("a", 1.0)
    metrics.record_metric("b", 2.0)
    assert metrics.get_metric("a") == 1.0
    assert metrics.get_metric("b") == 2.0


def test_record_metric_zero():
    metrics.record_metric("zero", 0.0)
    assert metrics.get_metric("zero") == 0.0


def test_record_metric_negative():
    metrics.record_metric("neg", -1.5)
    assert metrics.get_metric("neg") == -1.5


def test_init_tracing_default(capsys):
    tracing.init_tracing()
    captured = capsys.readouterr()
    assert "ai_system" in captured.out


def test_init_tracing_custom_service(capsys):
    tracing.init_tracing("my_service")
    captured = capsys.readouterr()
    assert "my_service" in captured.out


def test_init_tracing_noop_does_not_raise():
    tracing.init_tracing("test_service")

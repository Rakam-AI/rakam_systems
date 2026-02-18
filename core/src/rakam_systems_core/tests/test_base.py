import pytest
import time

from rakam_systems_core.base import BaseComponent




class DummyComponent(BaseComponent):
    def setup(self):
        super().setup()
        self.resource = "ready"

    def shutdown(self):
        super().shutdown()
        self.resource = None

    def run(self, x, y=0):
        return x + y

    def multiply(self, x, y):
        return x * y

    def fail(self):
        raise ValueError("boom")




def test_setup_and_shutdown():
    comp = DummyComponent("dummy")

    assert comp.initialized is False

    comp.setup()
    assert comp.initialized is True
    assert comp.resource == "ready"

    comp.shutdown()
    assert comp.initialized is False
    assert comp.resource is None


def test_call_auto_setup():
    comp = DummyComponent("dummy")

    assert comp.initialized is False
    result = comp(2, y=3)

    assert result == 5
    assert comp.initialized is True



def test_context_manager_calls_setup_and_shutdown():
    comp = DummyComponent("dummy")

    with comp as c:
        assert c.initialized is True
        assert c.resource == "ready"

    assert comp.initialized is False
    assert comp.resource is None


def test_context_manager_shutdown_on_exception():
    comp = DummyComponent("dummy")

    with pytest.raises(RuntimeError):
        with comp:
            raise RuntimeError("failure inside context")

    assert comp.initialized is False



def test_timed_returns_output_and_duration():
    comp = DummyComponent("dummy")

    def slow_add(x, y):
        time.sleep(0.01)
        return x + y

    out, duration = comp.timed(slow_add, 2, 3)

    assert out == 5
    assert duration >= 0.01



def test_evaluate_success_case():
    comp = DummyComponent("dummy")

    results = comp.evaluate(
        test_cases={
            "run": [
                {"args": [2], "kwargs": {"y": 3}, "expected": 5}
            ]
        },
        verbose=False,
    )

    assert "run" in results
    case = results["run"][0]

    assert case["success"] is True
    assert case["output"] == 5
    assert case["expected"] == 5
    assert "time" in case


def test_evaluate_with_metric():
    comp = DummyComponent("dummy")

    def metric(out, expected):
        return 1.0 if out == expected else 0.0

    results = comp.evaluate(
        test_cases={
            "run": [
                {"args": [2], "kwargs": {"y": 3}, "expected": 5}
            ]
        },
        metric_fn=metric,
        verbose=False,
    )

    case = results["run"][0]
    assert case["score"] == 1.0


def test_evaluate_failure_case():
    comp = DummyComponent("dummy")

    results = comp.evaluate(
        methods=["fail"],
        test_cases={
            "fail": [
                {"args": [], "kwargs": {}}
            ]
        },
        verbose=False,
    )

    case = results["fail"][0]
    assert case["success"] is False
    assert "boom" in case["error"]
    assert "traceback" in case


def test_evaluate_invalid_method():
    comp = DummyComponent("dummy")

    with pytest.raises(AttributeError):
        comp.evaluate(methods=["does_not_exist"], verbose=False)


def test_evaluate_non_callable_attribute():
    comp = DummyComponent("dummy")
    comp.not_callable = 123

    with pytest.raises(TypeError):
        comp.evaluate(methods=["not_callable"], verbose=False)


def test_evaluate_multiple_methods():
    comp = DummyComponent("dummy")

    results = comp.evaluate(
        methods=["run", "multiply"],
        test_cases={
            "run": [{"args": [1], "kwargs": {"y": 2}, "expected": 3}],
            "multiply": [{"args": [2, 3], "expected": 6}],
        },
        verbose=False,
    )

    assert results["run"][0]["success"] is True
    assert results["multiply"][0]["success"] is True

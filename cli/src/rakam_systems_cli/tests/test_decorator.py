import os
from pathlib import Path
from typing import List

import pytest

from rakam_eval_sdk.client import DeepEvalClient
from rakam_systems_cli.decorators import eval_run
from rakam_systems_cli.utils.decorator_utils import find_decorated_functions


@pytest.fixture
def client() -> DeepEvalClient:
    return DeepEvalClient(base_url="http://testserver", api_token="testtoken")


class FakeCPUTimes:
    def __init__(self, user: float = 1.0, system: float = 1.0) -> None:
        self.user: float = user
        self.system: float = system


class FakeMemInfo:
    def __init__(self, rss: int) -> None:
        self.rss: int = rss


class FakeProcess:
    def __init__(self) -> None:
        self._cpu_calls: int = 0
        self._mem_calls: int = 0

    def cpu_times(self) -> FakeCPUTimes:
        self._cpu_calls += 1
        # simulate CPU usage increase
        return FakeCPUTimes(
            user=1.0 + self._cpu_calls,
            system=1.0,
        )

    def memory_info(self) -> FakeMemInfo:
        self._mem_calls += 1
        return FakeMemInfo(rss=100_000_000 + (self._mem_calls * 10_000))


@pytest.fixture(autouse=True)
def patch_psutil(
    monkeypatch: pytest.MonkeyPatch,
) -> FakeProcess:
    fake_process: FakeProcess = FakeProcess()

    monkeypatch.setattr(
        "rakam_eval_sdk.decorators.psutil.Process",
        lambda pid: fake_process,
    )
    monkeypatch.setattr(os, "getpid", lambda: 123)
    return fake_process


def test_eval_run_basic(capsys: pytest.CaptureFixture[str]) -> None:
    @eval_run
    def add(a: int, b: int) -> int:
        return a + b

    result: int = add(2, 3)

    assert result == 5

    out: str = capsys.readouterr().out
    assert "[eval_run]" in out
    assert "add" in out
    assert "time=" in out
    assert "cpu=" in out
    assert "mem_delta=" in out


def test_eval_run_with_parentheses(capsys: pytest.CaptureFixture[str]) -> None:
    @eval_run()
    def mul(a: int, b: int) -> int:
        return a * b

    result: int = mul(3, 4)

    assert result == 12
    assert "[eval_run]" in capsys.readouterr().out


def test_find_decorated_functions(tmp_path: Path) -> None:
    code = """
from rakam_eval_sdk.decorators import eval_run

@eval_run
def foo():
    pass

@eval_run()
def bar():
    pass

async def baz():
    pass

@other
def nope():
    pass
"""
    file = tmp_path / "test_mod.py"
    file.write_text(code)

    result: List[str] = find_decorated_functions(file, "eval_run")

    assert set(result) == {"foo", "bar"}

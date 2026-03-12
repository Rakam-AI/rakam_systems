from pathlib import Path
from types import ModuleType
from typing import List

import pytest

from rakam_systems_cli.utils.decorator_utils import (
    find_decorated_functions,
    load_module_from_path,
)


def test_load_module_from_path_success(tmp_path: Path) -> None:
    file = tmp_path / "mod.py"
    file.write_text(
        """
x = 42
def hello():
    return "world"
"""
    )

    module: ModuleType = load_module_from_path(file)

    assert module.x == 42
    assert module.hello() == "world"


def test_load_module_from_path_invalid(tmp_path: Path) -> None:
    file: Path = tmp_path / "broken.py"
    file.write_text("def foo(")  # invalid syntax

    with pytest.raises(SyntaxError):
        load_module_from_path(file)


# ---------------------------------------------------------------------------
# find_decorated_functions
# ---------------------------------------------------------------------------


def test_find_decorated_functions_simple_decorator(tmp_path: Path) -> None:
    file = tmp_path / "eval_mod.py"
    file.write_text(
        """
from rakam_systems_cli.decorators import eval_run

@eval_run
def my_eval():
    pass

def not_decorated():
    pass
"""
    )
    results: List[str] = find_decorated_functions(file, "eval_run")
    assert "my_eval" in results
    assert "not_decorated" not in results


def test_find_decorated_functions_attribute_decorator(tmp_path: Path) -> None:
    file = tmp_path / "mod.py"
    file.write_text(
        """
import decorators

@decorators.eval_run
def decorated_func():
    pass
"""
    )
    results: List[str] = find_decorated_functions(file, "eval_run")
    assert "decorated_func" in results


def test_find_decorated_functions_call_decorator(tmp_path: Path) -> None:
    file = tmp_path / "mod.py"
    file.write_text(
        """
@eval_run(name="test")
def parameterized():
    pass
"""
    )
    results: List[str] = find_decorated_functions(file, "eval_run")
    assert "parameterized" in results


def test_find_decorated_functions_no_matches(tmp_path: Path) -> None:
    file = tmp_path / "mod.py"
    file.write_text(
        """
def plain_function():
    pass

@other_decorator
def other_func():
    pass
"""
    )
    results: List[str] = find_decorated_functions(file, "eval_run")
    assert results == []


def test_find_decorated_functions_multiple_functions(tmp_path: Path) -> None:
    file = tmp_path / "mod.py"
    file.write_text(
        """
@eval_run
def eval_one():
    pass

@eval_run
def eval_two():
    pass

def ignored():
    pass
"""
    )
    results: List[str] = find_decorated_functions(file, "eval_run")
    assert "eval_one" in results
    assert "eval_two" in results
    assert "ignored" not in results
    assert len(results) == 2


def test_find_decorated_functions_async_function(tmp_path: Path) -> None:
    file = tmp_path / "mod.py"
    file.write_text(
        """
@eval_run
async def async_eval():
    pass
"""
    )
    results: List[str] = find_decorated_functions(file, "eval_run")
    assert "async_eval" in results


def test_find_decorated_functions_empty_file(tmp_path: Path) -> None:
    file = tmp_path / "empty.py"
    file.write_text("")
    results: List[str] = find_decorated_functions(file, "eval_run")
    assert results == []


def test_find_decorated_functions_different_decorator_name(tmp_path: Path) -> None:
    file = tmp_path / "mod.py"
    file.write_text(
        """
@my_decorator
def func_a():
    pass

@other_decorator
def func_b():
    pass
"""
    )
    results_a: List[str] = find_decorated_functions(file, "my_decorator")
    results_b: List[str] = find_decorated_functions(file, "other_decorator")
    assert results_a == ["func_a"]
    assert results_b == ["func_b"]

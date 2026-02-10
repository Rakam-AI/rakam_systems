from pathlib import Path
from types import ModuleType

import pytest

from rakam_cli.utils.decorator_utils import load_module_from_path


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

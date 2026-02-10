import ast
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import List


class DecoratedFunctionVisitor(ast.NodeVisitor):
    def __init__(self, decorator_name: str):
        self.decorator_name = decorator_name
        self.results: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for deco in node.decorator_list:
            if self._matches(deco):
                self.results.append(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        for deco in node.decorator_list:
            if self._matches(deco):
                self.results.append(node.name)
        self.generic_visit(node)

    def _matches(self, deco: ast.expr) -> bool:
        # @deco
        if isinstance(deco, ast.Name):
            return deco.id == self.decorator_name

        # @module.deco
        if isinstance(deco, ast.Attribute):
            return deco.attr == self.decorator_name

        # @deco(...)
        if isinstance(deco, ast.Call):
            return self._matches(deco.func)

        return False


def find_decorated_functions(
    file_path: Path,
    decorator_name: str,
) -> List[str]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    visitor = DecoratedFunctionVisitor(decorator_name)
    visitor.visit(tree)
    return visitor.results


def load_module_from_path(file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

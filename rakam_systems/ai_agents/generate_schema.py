import ast
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None


def get_class_bases(node: ast.ClassDef) -> List[str]:
    """Return base class names for a given class node."""
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            if isinstance(base.value, ast.Name):
                bases.append(f"{base.value.id}.{base.attr}")
            else:
                bases.append(base.attr)
        elif isinstance(base, ast.Subscript):
            if isinstance(base.value, ast.Name):
                bases.append(base.value.id)
    return bases


def extract_class_fields(node: ast.ClassDef) -> Dict[str, str]:
    """Extract fields from annotated assignments and regular assignments."""
    fields = {}
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            field_name = stmt.target.id
            type_name = getattr(stmt.annotation, 'id',
                                str(ast.dump(stmt.annotation)))
            fields[field_name] = type_name
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                fields[target.id] = "Any"
    return fields


def detect_class_type(bases: List[str], decorators: List[ast.expr]) -> str:
    """Determine whether the class is pydantic, dataclass, or normal."""
    if any("BaseModel" in b for b in bases):
        return "pydantic"
    if any(isinstance(d, ast.Name) and d.id == "dataclass" for d in decorators):
        return "dataclass"
    return "normal"


def find_target_classes(tree: ast.Module, targets: List[str]) -> List[ast.ClassDef]:
    """Return classes that inherit from target base classes."""
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = get_class_bases(node)
            if any(base in targets for base in bases):
                result.append(node)
    return result


def generate_openapi_schema(class_name: str, fields: Dict[str, str], class_type: str) -> Dict[str, Any]:
    """Fallback OpenAPI schema generator (for non-Pydantic classes)."""
    properties = {}
    for name, type_ in fields.items():
        openapi_type = "string"
        if type_ in {"int", "float"}:
            openapi_type = "number"
        elif type_ == "bool":
            openapi_type = "boolean"
        elif type_ in {"dict", "Dict"}:
            openapi_type = "object"
        elif type_ in {"list", "List"}:
            openapi_type = "array"

        properties[name] = {"type": openapi_type}

    return {
        "title": class_name,
        "type": "object",
        "properties": properties,
        "x-class-type": class_type,
    }


def load_module(filepath: str):
    """Dynamically import a module by filepath."""
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot import {filepath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=filepath)
    target_bases = ["AgentInput", "AgentOutput"]

    classes = find_target_classes(tree, target_bases)
    schemas = {}

    # try importing module for real Pydantic class inspection
    try:
        module = load_module(filepath)
    except Exception as e:
        print(f"Warning: could not import module ({e}), fallback to AST only.")
        module = None

    for cls in classes:
        class_name = cls.name
        bases = get_class_bases(cls)
        class_type = detect_class_type(bases, cls.decorator_list)

        # try real introspection for Pydantic models
        if module and class_type == "pydantic" and hasattr(module, class_name):
            cls_obj = getattr(module, class_name)
            if BaseModel and issubclass(cls_obj, BaseModel):
                try:
                    schema = cls_obj.model_json_schema()
                    schema["x-class-type"] = "pydantic"
                    schemas[class_name] = schema
                    continue
                except Exception as e:
                    print(
                        f"Warning: failed to get model_json_schema for {class_name}: {e}")

        # fallback to AST-based schema generation
        fields = extract_class_fields(cls)
        schemas[class_name] = generate_openapi_schema(
            class_name, fields, class_type)

    openapi = {
        "openapi": "3.1.0",
        "info": {"title": "Auto-Generated Schema", "version": "1.0.0"},
        "components": {"schemas": schemas},
    }

    print(json.dumps(openapi, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_schema.py <path_to_file.py>")
        sys.exit(1)
    main(sys.argv[1])

"""Tests for config_loader.py ConfigurationLoader."""
from pathlib import Path

import pytest
import yaml

from rakam_systems_core.config_loader import ConfigurationLoader
from rakam_systems_core.config_schema import (
    ConfigFileSchema,
    OutputFieldSchema,
    OutputTypeSchema,
    PromptConfigSchema,
    ToolConfigSchema,
    ToolMode,
)


@pytest.fixture
def loader() -> ConfigurationLoader:
    return ConfigurationLoader()


@pytest.fixture
def simple_config_dict() -> dict:
    return {
        "prompts": {
            "p1": {
                "name": "p1",
                "system_prompt": "You are helpful.",
            }
        },
        "tools": {
            "get_cwd": {
                "name": "get_cwd",
                "type": "direct",
                "description": "Get current directory",
                "module": "os",
                "function": "getcwd",
            }
        },
        "agents": {},
    }


# ---------------------------------------------------------------------------
# load_from_dict
# ---------------------------------------------------------------------------


class TestLoadFromDict:
    def test_basic_load(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        assert isinstance(config, ConfigFileSchema)
        assert "p1" in config.prompts
        assert "get_cwd" in config.tools

    def test_sets_internal_config(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        assert loader.config is config

    def test_empty_config(self, loader):
        config = loader.load_from_dict({})
        assert isinstance(config, ConfigFileSchema)
        assert config.agents == {}


# ---------------------------------------------------------------------------
# load_from_yaml
# ---------------------------------------------------------------------------


class TestLoadFromYaml:
    def test_basic_load(self, loader, simple_config_dict, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(simple_config_dict))
        config = loader.load_from_yaml(str(yaml_file))
        assert isinstance(config, ConfigFileSchema)
        assert "p1" in config.prompts

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_from_yaml("/nonexistent/path/config.yaml")

    def test_sets_internal_config(self, loader, simple_config_dict, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(simple_config_dict))
        config = loader.load_from_yaml(str(yaml_file))
        assert loader.config is config


# ---------------------------------------------------------------------------
# _generate_schema
# ---------------------------------------------------------------------------


class TestGenerateSchema:
    def test_simple_string_function(self, loader):
        def my_func(name: str) -> None:
            pass

        schema = loader._generate_schema(my_func)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "name" in schema["required"]
        assert schema["properties"]["name"]["type"] == "string"

    def test_int_type(self, loader):
        def my_func(count: int) -> None:
            pass

        schema = loader._generate_schema(my_func)
        assert schema["properties"]["count"]["type"] == "number"

    def test_float_type(self, loader):
        def my_func(score: float) -> None:
            pass

        schema = loader._generate_schema(my_func)
        assert schema["properties"]["score"]["type"] == "number"

    def test_bool_type(self, loader):
        def my_func(flag: bool) -> None:
            pass

        schema = loader._generate_schema(my_func)
        assert schema["properties"]["flag"]["type"] == "boolean"

    def test_list_type(self, loader):
        def my_func(items: list) -> None:
            pass

        schema = loader._generate_schema(my_func)
        assert schema["properties"]["items"]["type"] == "array"

    def test_dict_type(self, loader):
        def my_func(data: dict) -> None:
            pass

        schema = loader._generate_schema(my_func)
        assert schema["properties"]["data"]["type"] == "object"

    def test_function_with_defaults_not_in_required(self, loader):
        def my_func(x: str, y: int = 5) -> None:
            pass

        schema = loader._generate_schema(my_func)
        assert "x" in schema["required"]
        assert "y" not in schema["required"]

    def test_skips_self_cls_ctx_context(self, loader):
        def my_func(self, cls, ctx, context, arg: str) -> None:
            pass

        schema = loader._generate_schema(my_func)
        assert "self" not in schema["properties"]
        assert "cls" not in schema["properties"]
        assert "ctx" not in schema["properties"]
        assert "context" not in schema["properties"]
        assert "arg" in schema["properties"]

    def test_no_annotation_defaults_to_string(self, loader):
        def my_func(arg) -> None:
            pass

        schema = loader._generate_schema(my_func)
        assert schema["properties"]["arg"]["type"] == "string"

    def test_additional_properties_false(self, loader):
        def my_func(x: str) -> None:
            pass

        schema = loader._generate_schema(my_func)
        assert schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# _load_function
# ---------------------------------------------------------------------------


class TestLoadFunction:
    def test_load_existing_function(self, loader):
        func = loader._load_function("os.path", "join")
        assert callable(func)

    def test_import_error(self, loader):
        with pytest.raises(ImportError):
            loader._load_function("nonexistent.module.xyz", "func")

    def test_attribute_error(self, loader):
        with pytest.raises(AttributeError):
            loader._load_function("os", "nonexistent_function_xyz_abc")

    def test_not_callable_raises_attribute_error(self, loader):
        # os.sep is a string, not callable
        with pytest.raises((AttributeError, ValueError)):
            loader._load_function("os", "sep")


# ---------------------------------------------------------------------------
# _load_class
# ---------------------------------------------------------------------------


class TestLoadClass:
    def test_load_existing_class(self, loader):
        cls = loader._load_class("pathlib.Path")
        assert cls is Path

    def test_import_error(self, loader):
        with pytest.raises(ImportError):
            loader._load_class("nonexistent.module.MyClass")

    def test_attribute_error(self, loader):
        with pytest.raises(AttributeError):
            loader._load_class("pathlib.NonExistentClassXyz")

    def test_invalid_path_format(self, loader):
        with pytest.raises(ValueError):
            loader._load_class("InvalidPathNoModule")

    def test_not_a_class_raises_value_error(self, loader):
        # os.sep is a string, not a class
        with pytest.raises(ValueError):
            loader._load_class("os.sep")


# ---------------------------------------------------------------------------
# resolve_prompt_config
# ---------------------------------------------------------------------------


class TestResolvePromptConfig:
    def test_by_name(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        prompt = loader.resolve_prompt_config("p1", config)
        assert prompt.name == "p1"
        assert prompt.system_prompt == "You are helpful."

    def test_by_schema_instance(self, loader):
        schema = PromptConfigSchema(name="direct", system_prompt="direct prompt")
        result = loader.resolve_prompt_config(schema)
        assert result is schema

    def test_missing_prompt_raises(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        with pytest.raises(ValueError, match="not found"):
            loader.resolve_prompt_config("nonexistent", config)

    def test_no_config_raises(self, loader):
        with pytest.raises(ValueError, match="No configuration"):
            loader.resolve_prompt_config("p1")

    def test_uses_loaded_config(self, loader, simple_config_dict):
        loader.load_from_dict(simple_config_dict)
        # Should use loader.config when no config arg provided
        prompt = loader.resolve_prompt_config("p1")
        assert prompt.name == "p1"


# ---------------------------------------------------------------------------
# resolve_tools
# ---------------------------------------------------------------------------


class TestResolveTools:
    def test_by_name(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        tools = loader.resolve_tools(["get_cwd"], config)
        assert len(tools) == 1
        assert tools[0].name == "get_cwd"

    def test_by_schema_instance(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        schema = ToolConfigSchema(
            name="inline_tool",
            type=ToolMode.DIRECT,
            description="inline",
            module="os",
            function="getcwd",
        )
        tools = loader.resolve_tools([schema], config)
        assert tools[0] is schema

    def test_missing_tool_raises(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        with pytest.raises(ValueError, match="not found"):
            loader.resolve_tools(["nonexistent_tool"], config)

    def test_no_config_raises(self, loader):
        with pytest.raises(ValueError):
            loader.resolve_tools(["t"])

    def test_empty_list(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        tools = loader.resolve_tools([], config)
        assert tools == []

    def test_invalid_ref_raises(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        with pytest.raises(ValueError):
            loader.resolve_tools([123], config)  # invalid type


# ---------------------------------------------------------------------------
# get_tool_registry
# ---------------------------------------------------------------------------


class TestGetToolRegistry:
    def test_direct_tool_registration(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        registry = loader.get_tool_registry(config)
        tools = registry.get_all_tools()
        names = [t.name for t in tools]
        assert "get_cwd" in names

    def test_returns_cached_registry(self, loader, simple_config_dict):
        config = loader.load_from_dict(simple_config_dict)
        registry1 = loader.get_tool_registry(config)
        registry2 = loader.get_tool_registry(config)
        assert registry1 is registry2

    def test_no_config_raises(self, loader):
        with pytest.raises(ValueError):
            loader.get_tool_registry()

    def test_uses_loaded_config(self, loader, simple_config_dict):
        loader.load_from_dict(simple_config_dict)
        registry = loader.get_tool_registry()
        tools = registry.get_all_tools()
        assert len(tools) > 0


# ---------------------------------------------------------------------------
# _create_output_type_from_schema
# ---------------------------------------------------------------------------


class TestCreateOutputTypeFromSchema:
    def test_simple_str_field(self, loader):
        schema = OutputTypeSchema(
            name="MyModel",
            fields={
                "answer": OutputFieldSchema(type="str", description="The answer"),
            },
        )
        model_class = loader._create_output_type_from_schema(schema)
        assert model_class.__name__ == "MyModel"
        instance = model_class(answer="hello")
        assert instance.answer == "hello"

    def test_field_with_default(self, loader):
        schema = OutputTypeSchema(
            name="WithDefault",
            fields={
                "answer": OutputFieldSchema(type="str", description="ans"),
                "count": OutputFieldSchema(type="int", description="cnt", default=0),
            },
        )
        model_class = loader._create_output_type_from_schema(schema)
        instance = model_class(answer="ok")
        assert instance.count == 0

    def test_all_primitive_types(self, loader):
        schema = OutputTypeSchema(
            name="AllTypes",
            fields={
                "s": OutputFieldSchema(type="str", description="s"),
                "i": OutputFieldSchema(type="int", description="i", default=1),
                "f": OutputFieldSchema(type="float", description="f", default=0.0),
                "b": OutputFieldSchema(type="bool", description="b", default=False),
            },
        )
        model_class = loader._create_output_type_from_schema(schema)
        instance = model_class(s="test")
        assert instance.i == 1
        assert instance.f == 0.0
        assert instance.b is False

    def test_list_default_factory(self, loader):
        schema = OutputTypeSchema(
            name="WithList",
            fields={
                "items": OutputFieldSchema(
                    type="list",
                    description="items",
                    default_factory="list",
                ),
            },
        )
        model_class = loader._create_output_type_from_schema(schema)
        instance = model_class()
        assert instance.items == []

    def test_dict_default_factory(self, loader):
        schema = OutputTypeSchema(
            name="WithDict",
            fields={
                "data": OutputFieldSchema(
                    type="dict",
                    description="data",
                    default_factory="dict",
                ),
            },
        )
        model_class = loader._create_output_type_from_schema(schema)
        instance = model_class()
        assert instance.data == {}

    def test_optional_field_defaults_to_none(self, loader):
        schema = OutputTypeSchema(
            name="Optional",
            fields={
                "required_field": OutputFieldSchema(type="str", description="req"),
                "optional_field": OutputFieldSchema(
                    type="str", description="opt", required=False
                ),
            },
        )
        model_class = loader._create_output_type_from_schema(schema)
        instance = model_class(required_field="yes")
        assert instance.optional_field is None

    def test_type_aliases(self, loader):
        schema = OutputTypeSchema(
            name="Aliases",
            fields={
                "string_field": OutputFieldSchema(
                    type="string", description="s", default=""
                ),
                "integer_field": OutputFieldSchema(
                    type="integer", description="i", default=0
                ),
                "number_field": OutputFieldSchema(
                    type="number", description="n", default=0.0
                ),
                "boolean_field": OutputFieldSchema(
                    type="boolean", description="b", default=False
                ),
                "array_field": OutputFieldSchema(
                    type="array", description="a", default_factory="list"
                ),
                "object_field": OutputFieldSchema(
                    type="object", description="o", default_factory="dict"
                ),
            },
        )
        model_class = loader._create_output_type_from_schema(schema)
        instance = model_class()
        assert instance.string_field == ""


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config(self, loader):
        loader.load_from_dict({"agents": {}, "tools": {}, "prompts": {}})
        is_valid, errors = loader.validate_config()
        assert is_valid is True
        assert errors == []

    def test_no_config_loaded(self, loader):
        is_valid, errors = loader.validate_config()
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_yaml_file(self, loader, simple_config_dict, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(simple_config_dict))
        is_valid, errors = loader.validate_config(str(yaml_file))
        assert is_valid is True

    def test_invalid_yaml_file_path(self, loader):
        is_valid, errors = loader.validate_config("/nonexistent/config.yaml")
        assert is_valid is False
        assert len(errors) > 0

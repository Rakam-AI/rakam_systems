"""
Tests for the configuration system.

Tests cover:
- Configuration schema validation
- Configuration loading from YAML and dict
- Agent creation from configuration
- Tool and prompt resolution
- Configuration validation
"""
import pytest
from pathlib import Path
import tempfile
import yaml

from ai_core.config_schema import (
    ConfigFileSchema,
    AgentConfigSchema,
    ToolConfigSchema,
    ModelConfigSchema,
    PromptConfigSchema,
    ToolMode,
)
from ai_core.config_loader import ConfigurationLoader


class TestConfigSchemas:
    """Test Pydantic configuration schemas."""
    
    def test_model_config_schema(self):
        """Test ModelConfigSchema validation."""
        config = ModelConfigSchema(
            model="openai:gpt-4o",
            temperature=0.7,
            max_tokens=1000,
            parallel_tool_calls=True,
        )
        
        assert config.model == "openai:gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.parallel_tool_calls is True
    
    def test_model_config_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        ModelConfigSchema(model="test", temperature=0.0)
        ModelConfigSchema(model="test", temperature=1.0)
        ModelConfigSchema(model="test", temperature=2.0)
        
        # Invalid temperatures
        with pytest.raises(ValueError):
            ModelConfigSchema(model="test", temperature=-0.1)
        
        with pytest.raises(ValueError):
            ModelConfigSchema(model="test", temperature=2.1)
    
    def test_prompt_config_schema(self):
        """Test PromptConfigSchema."""
        config = PromptConfigSchema(
            name="test_prompt",
            system_prompt="You are a helpful assistant.",
            description="Test prompt",
            skills=["skill1", "skill2"],
            tags=["tag1"],
        )
        
        assert config.name == "test_prompt"
        assert "helpful assistant" in config.system_prompt
        assert len(config.skills) == 2
    
    def test_tool_config_direct(self):
        """Test ToolConfigSchema for direct tools."""
        config = ToolConfigSchema(
            name="test_tool",
            type=ToolMode.DIRECT,
            description="Test tool",
            module="test.module",
            function="test_func",
            category="test",
            tags=["tag1"],
        )
        
        assert config.type == ToolMode.DIRECT
        assert config.module == "test.module"
        assert config.function == "test_func"
    
    def test_tool_config_mcp(self):
        """Test ToolConfigSchema for MCP tools."""
        config = ToolConfigSchema(
            name="test_tool",
            type=ToolMode.MCP,
            description="Test tool",
            mcp_server="test_server",
            mcp_tool_name="remote_tool",
        )
        
        assert config.type == ToolMode.MCP
        assert config.mcp_server == "test_server"
        assert config.mcp_tool_name == "remote_tool"
    
    def test_tool_config_validation_direct(self):
        """Test that direct tools require module."""
        with pytest.raises(ValueError):
            ToolConfigSchema(
                name="test",
                type=ToolMode.DIRECT,
                description="Test",
                # Missing module
            )
    
    def test_tool_config_validation_mcp(self):
        """Test that MCP tools require mcp_server."""
        with pytest.raises(ValueError):
            ToolConfigSchema(
                name="test",
                type=ToolMode.MCP,
                description="Test",
                # Missing mcp_server
            )
    
    def test_agent_config_schema(self):
        """Test AgentConfigSchema."""
        config = AgentConfigSchema(
            name="test_agent",
            description="Test agent",
            model_config=ModelConfigSchema(
                model="openai:gpt-4o",
                temperature=0.7,
            ),
            prompt_config="test_prompt",
            tools=["tool1", "tool2"],
            enable_tracking=True,
            tracking_output_dir="./tracking",
        )
        
        assert config.name == "test_agent"
        assert config.model_config.model == "openai:gpt-4o"
        assert config.prompt_config == "test_prompt"
        assert len(config.tools) == 2
        assert config.enable_tracking is True
    
    def test_config_file_schema(self):
        """Test complete ConfigFileSchema."""
        config = ConfigFileSchema(
            version="1.0",
            prompts={
                "prompt1": PromptConfigSchema(
                    name="prompt1",
                    system_prompt="Test",
                )
            },
            tools={
                "tool1": ToolConfigSchema(
                    name="tool1",
                    type=ToolMode.DIRECT,
                    description="Test",
                    module="test",
                    function="func",
                )
            },
            agents={
                "agent1": AgentConfigSchema(
                    name="agent1",
                    model_config=ModelConfigSchema(model="test"),
                    prompt_config="prompt1",
                    tools=["tool1"],
                )
            }
        )
        
        assert config.version == "1.0"
        assert "prompt1" in config.prompts
        assert "tool1" in config.tools
        assert "agent1" in config.agents


class TestConfigurationLoader:
    """Test ConfigurationLoader functionality."""
    
    def create_test_config(self) -> dict:
        """Create a minimal test configuration."""
        return {
            "version": "1.0",
            "prompts": {
                "test_prompt": {
                    "name": "test_prompt",
                    "system_prompt": "You are a test assistant.",
                    "skills": ["testing"],
                }
            },
            "tools": {
                "test_tool": {
                    "name": "test_tool",
                    "type": "direct",
                    "module": "builtins",
                    "function": "len",
                    "description": "Test tool",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "obj": {"type": "object"}
                        },
                    }
                }
            },
            "agents": {
                "test_agent": {
                    "name": "test_agent",
                    "description": "Test agent",
                    "model_config": {
                        "model": "openai:gpt-4o-mini",
                        "temperature": 0.7,
                        "parallel_tool_calls": True,
                    },
                    "prompt_config": "test_prompt",
                    "tools": ["test_tool"],
                    "enable_tracking": False,
                    "stateful": False,
                }
            }
        }
    
    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        loader = ConfigurationLoader()
        config_dict = self.create_test_config()
        
        config = loader.load_from_dict(config_dict)
        
        assert config.version == "1.0"
        assert "test_prompt" in config.prompts
        assert "test_tool" in config.tools
        assert "test_agent" in config.agents
    
    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        loader = ConfigurationLoader()
        config_dict = self.create_test_config()
        
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = loader.load_from_yaml(temp_path)
            
            assert config.version == "1.0"
            assert "test_prompt" in config.prompts
            assert "test_tool" in config.tools
            assert "test_agent" in config.agents
        finally:
            Path(temp_path).unlink()
    
    def test_resolve_prompt_config(self):
        """Test prompt configuration resolution."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(self.create_test_config())
        
        prompt = loader.resolve_prompt_config("test_prompt", config)
        
        assert prompt.name == "test_prompt"
        assert "test assistant" in prompt.system_prompt
    
    def test_resolve_tools(self):
        """Test tool resolution."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(self.create_test_config())
        
        tools = loader.resolve_tools(["test_tool"], config)
        
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
    
    def test_resolve_nonexistent_prompt(self):
        """Test resolving non-existent prompt raises error."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(self.create_test_config())
        
        with pytest.raises(ValueError, match="not found"):
            loader.resolve_prompt_config("nonexistent", config)
    
    def test_resolve_nonexistent_tool(self):
        """Test resolving non-existent tool raises error."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(self.create_test_config())
        
        with pytest.raises(ValueError, match="not found"):
            loader.resolve_tools(["nonexistent"], config)
    
    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        loader = ConfigurationLoader()
        config_dict = self.create_test_config()
        loader.load_from_dict(config_dict)
        
        is_valid, errors = loader.validate_config()
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_config_invalid_prompt_ref(self):
        """Test validation catches invalid prompt reference."""
        loader = ConfigurationLoader()
        config_dict = self.create_test_config()
        
        # Reference non-existent prompt
        config_dict["agents"]["test_agent"]["prompt_config"] = "nonexistent"
        
        loader.load_from_dict(config_dict)
        is_valid, errors = loader.validate_config()
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_validate_config_invalid_tool_ref(self):
        """Test validation catches invalid tool reference."""
        loader = ConfigurationLoader()
        config_dict = self.create_test_config()
        
        # Reference non-existent tool
        config_dict["agents"]["test_agent"]["tools"] = ["nonexistent"]
        
        loader.load_from_dict(config_dict)
        is_valid, errors = loader.validate_config()
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_load_function(self):
        """Test dynamic function loading."""
        loader = ConfigurationLoader()
        
        # Load built-in function
        func = loader._load_function("builtins", "len")
        
        assert callable(func)
        assert func([1, 2, 3]) == 3
    
    def test_load_nonexistent_module(self):
        """Test loading non-existent module raises error."""
        loader = ConfigurationLoader()
        
        with pytest.raises(ImportError):
            loader._load_function("nonexistent.module", "func")
    
    def test_load_nonexistent_function(self):
        """Test loading non-existent function raises error."""
        loader = ConfigurationLoader()
        
        with pytest.raises(AttributeError):
            loader._load_function("builtins", "nonexistent_function")
    
    def test_generate_schema(self):
        """Test automatic schema generation."""
        loader = ConfigurationLoader()
        
        def test_func(x: int, y: str) -> str:
            return f"{x}: {y}"
        
        schema = loader._generate_schema(test_func)
        
        assert schema["type"] == "object"
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]
        assert schema["properties"]["x"]["type"] == "number"
        assert schema["properties"]["y"]["type"] == "string"
        assert set(schema["required"]) == {"x", "y"}


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def test_end_to_end_config_loading(self):
        """Test complete configuration loading flow."""
        config_dict = {
            "version": "1.0",
            "prompts": {
                "assistant": {
                    "name": "assistant",
                    "system_prompt": "You are helpful.",
                }
            },
            "tools": {},
            "agents": {
                "simple_agent": {
                    "name": "simple_agent",
                    "model_config": {
                        "model": "openai:gpt-4o-mini",
                        "temperature": 0.5,
                    },
                    "prompt_config": "assistant",
                    "tools": [],
                    "enable_tracking": False,
                }
            }
        }
        
        loader = ConfigurationLoader()
        config = loader.load_from_dict(config_dict)
        
        # Validate
        is_valid, errors = loader.validate_config()
        assert is_valid is True
        
        # Get tool registry
        registry = loader.get_tool_registry()
        assert registry is not None
        
        # Note: We don't create the actual agent here since it requires
        # pydantic_ai and other dependencies that may not be available in tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


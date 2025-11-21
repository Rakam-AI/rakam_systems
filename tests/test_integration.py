"""
Integration tests for the complete configuration and tracking system.

These tests verify that all components work together correctly.
"""
import pytest
import asyncio
from pathlib import Path
import tempfile
import yaml

from ai_core.config_loader import ConfigurationLoader
from ai_core.interfaces import ModelSettings


class TestCompleteIntegration:
    """Test complete configuration and tracking integration."""
    
    def create_test_config(self, tmpdir: str) -> dict:
        """Create a complete test configuration."""
        return {
            "version": "1.0",
            "global_settings": {
                "default_tracking_dir": tmpdir,
            },
            "prompts": {
                "test_assistant": {
                    "name": "test_assistant",
                    "system_prompt": "You are a test assistant that provides concise answers.",
                    "skills": ["Testing", "Validation"],
                    "tags": ["test"],
                }
            },
            "tools": {
                "format_currency": {
                    "name": "format_currency",
                    "type": "direct",
                    "module": "ai_agents.components.tools.example_tools",
                    "function": "format_currency",
                    "description": "Format a number as currency",
                    "category": "utility",
                    "tags": ["formatting"],
                    "schema": {
                        "type": "object",
                        "properties": {
                            "amount": {"type": "number"},
                            "currency": {"type": "string", "enum": ["USD", "EUR"]},
                        },
                        "required": ["amount"],
                        "additionalProperties": False,
                    }
                }
            },
            "agents": {
                "test_agent": {
                    "name": "test_agent",
                    "description": "Test agent with tracking",
                    "model_config": {
                        "model": "openai:gpt-4o-mini",
                        "temperature": 0.5,
                        "max_tokens": 500,
                        "parallel_tool_calls": True,
                    },
                    "prompt_config": "test_assistant",
                    "tools": ["format_currency"],
                    "enable_tracking": True,
                    "tracking_output_dir": str(Path(tmpdir) / "tracking"),
                    "stateful": False,
                }
            }
        }
    
    def test_load_validate_create(self):
        """Test: Load config -> Validate -> Create agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create configuration
            config_dict = self.create_test_config(tmpdir)
            
            # Save to YAML
            config_path = Path(tmpdir) / "test_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f)
            
            # Load configuration
            loader = ConfigurationLoader()
            config = loader.load_from_yaml(str(config_path))
            
            assert config.version == "1.0"
            assert "test_assistant" in config.prompts
            assert "format_currency" in config.tools
            assert "test_agent" in config.agents
            
            # Validate configuration
            is_valid, errors = loader.validate_config()
            
            assert is_valid is True, f"Config validation failed: {errors}"
            assert len(errors) == 0
            
            # Create agent (this will fail without pydantic_ai, but that's ok for test)
            try:
                agent = loader.create_agent("test_agent")
                
                # Verify agent properties
                assert agent.name == "test_agent"
                assert agent._tracking_enabled is True
                assert agent.model == "openai:gpt-4o-mini"
                
                print("✓ Agent created successfully with tracking enabled")
                
            except ImportError as e:
                # Expected if pydantic_ai not installed
                print(f"ℹ️  Skipping agent creation (pydantic_ai not available): {e}")
    
    def test_tool_registry_creation(self):
        """Test: Create tool registry from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = self.create_test_config(tmpdir)
            
            loader = ConfigurationLoader()
            loader.load_from_dict(config_dict)
            
            # Get tool registry
            registry = loader.get_tool_registry()
            
            assert registry is not None
            assert len(registry) == 1
            assert "format_currency" in registry
            
            # Get tool metadata
            tool_metadata = registry.get_tool("format_currency")
            
            assert tool_metadata is not None
            assert tool_metadata.name == "format_currency"
            assert tool_metadata.mode == "direct"
            assert tool_metadata.category == "utility"
    
    def test_multiple_agents(self):
        """Test: Create multiple agents from single config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = self.create_test_config(tmpdir)
            
            # Add another agent
            config_dict["agents"]["test_agent_2"] = {
                "name": "test_agent_2",
                "model_config": {
                    "model": "openai:gpt-4o",
                    "temperature": 0.7,
                },
                "prompt_config": "test_assistant",
                "tools": [],
                "enable_tracking": False,
            }
            
            loader = ConfigurationLoader()
            loader.load_from_dict(config_dict)
            
            # Validate
            is_valid, errors = loader.validate_config()
            assert is_valid is True
            
            # Try to create all agents
            try:
                agents = loader.create_all_agents()
                
                assert len(agents) == 2
                assert "test_agent" in agents
                assert "test_agent_2" in agents
                
                assert agents["test_agent"]._tracking_enabled is True
                assert agents["test_agent_2"]._tracking_enabled is False
                
                print("✓ Multiple agents created successfully")
                
            except ImportError:
                print("ℹ️  Skipping multiple agent creation (pydantic_ai not available)")
    
    def test_config_with_no_tools(self):
        """Test: Agent with no tools still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = {
                "version": "1.0",
                "prompts": {
                    "simple": {
                        "name": "simple",
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
                        "prompt_config": "simple",
                        "tools": [],
                        "enable_tracking": True,
                        "tracking_output_dir": tmpdir,
                    }
                }
            }
            
            loader = ConfigurationLoader()
            loader.load_from_dict(config_dict)
            
            is_valid, errors = loader.validate_config()
            assert is_valid is True
            
            # Get tool registry (should be empty)
            registry = loader.get_tool_registry()
            assert len(registry) == 0
    
    def test_invalid_references(self):
        """Test: Invalid references are caught by validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = self.create_test_config(tmpdir)
            
            # Break the prompt reference
            config_dict["agents"]["test_agent"]["prompt_config"] = "nonexistent_prompt"
            
            loader = ConfigurationLoader()
            loader.load_from_dict(config_dict)
            
            is_valid, errors = loader.validate_config()
            
            assert is_valid is False
            assert len(errors) > 0
            assert any("not found" in str(error).lower() for error in errors)
    
    def test_tool_loading(self):
        """Test: Tools are loaded correctly from modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = self.create_test_config(tmpdir)
            
            loader = ConfigurationLoader()
            loader.load_from_dict(config_dict)
            
            # Test function loading
            func = loader._load_function(
                "ai_agents.components.tools.example_tools",
                "format_currency"
            )
            
            assert callable(func)
            
            # Test the function works
            result = func(100, "USD")
            assert "$100.00" in result


class TestTrackingIntegration:
    """Test tracking system integration."""
    
    def test_tracking_manager_lifecycle(self):
        """Test: Complete tracking manager lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from ai_core.tracking import TrackingManager
            from ai_core.config_schema import MethodInputSchema, MethodOutputSchema
            from datetime import datetime
            
            manager = TrackingManager(output_dir=tmpdir)
            
            # Start session
            session_id = manager.start_session("test_agent")
            assert session_id in manager.sessions
            
            # Add records
            for i in range(3):
                input_data = MethodInputSchema(
                    timestamp=datetime.now(),
                    method_name="test_method",
                    agent_name="test_agent",
                    input_text=f"Test input {i}",
                    call_id=f"call-{i}",
                )
                output_data = MethodOutputSchema(
                    timestamp=datetime.now(),
                    method_name="test_method",
                    agent_name="test_agent",
                    output_text=f"Test output {i}",
                    duration_seconds=1.0 + i * 0.5,
                    success=True,
                    call_id=f"call-{i}",
                )
                
                manager.record_call(
                    agent_name="test_agent",
                    method_name="test_method",
                    input_data=input_data,
                    output_data=output_data,
                    session_id=session_id,
                )
            
            # End session
            manager.end_session(session_id)
            
            session = manager.get_session(session_id)
            assert session.total_calls == 3
            assert session.ended_at is not None
            
            # Export to CSV
            csv_path = manager.export_to_csv(session_id=session_id)
            assert csv_path.exists()
            
            # Verify CSV content
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 3
                for i, row in enumerate(rows):
                    assert row['call_id'] == f'call-{i}'
                    assert row['input_text'] == f'Test input {i}'
                    assert row['output_text'] == f'Test output {i}'
            
            # Export to JSON
            json_path = manager.export_to_json(session_id=session_id)
            assert json_path.exists()
            
            # Get statistics
            stats = manager.get_statistics(session_id)
            assert stats['total_calls'] == 3
            assert stats['successful_calls'] == 3
            assert stats['success_rate'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


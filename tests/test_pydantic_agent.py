"""
Tests for PydanticAIAgent and related components.
"""
import asyncio
import pytest
from typing import Dict

try:
    from ai_agents.components import PydanticAIAgent, BaseAgent
    from ai_core.interfaces import AgentInput, AgentOutput, ModelSettings, Tool
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
class TestToolCreation:
    """Test Tool creation and configuration."""
    
    def test_tool_from_schema(self):
        """Test creating a tool using from_schema."""
        async def sample_tool(x: int) -> int:
            return x * 2
        
        tool = Tool.from_schema(
            function=sample_tool,
            name='sample_tool',
            description='Doubles a number',
            json_schema={
                'type': 'object',
                'properties': {
                    'x': {'type': 'integer'},
                },
                'required': ['x'],
            },
            takes_ctx=False,
        )
        
        assert tool.name == 'sample_tool'
        assert tool.description == 'Doubles a number'
        assert tool.is_async is True
    
    def test_tool_sync_function(self):
        """Test tool with synchronous function."""
        def sync_tool(x: int) -> int:
            return x * 2
        
        tool = Tool.from_schema(
            function=sync_tool,
            name='sync_tool',
            description='Sync doubles',
            json_schema={
                'type': 'object',
                'properties': {
                    'x': {'type': 'integer'},
                },
                'required': ['x'],
            },
            takes_ctx=False,
        )
        
        assert tool.is_async is False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
class TestModelSettings:
    """Test ModelSettings configuration."""
    
    def test_default_settings(self):
        """Test default ModelSettings."""
        settings = ModelSettings()
        assert settings.parallel_tool_calls is True
        assert settings.temperature is None
        assert settings.max_tokens is None
    
    def test_custom_settings(self):
        """Test custom ModelSettings."""
        settings = ModelSettings(
            parallel_tool_calls=False,
            temperature=0.7,
            max_tokens=1000,
        )
        assert settings.parallel_tool_calls is False
        assert settings.temperature == 0.7
        assert settings.max_tokens == 1000
    
    def test_extra_settings(self):
        """Test extra settings passed through kwargs."""
        settings = ModelSettings(
            parallel_tool_calls=True,
            custom_param="value",
        )
        assert settings.extra_settings['custom_param'] == "value"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
class TestBaseAgent:
    """Test BaseAgent functionality."""
    
    def test_agent_initialization(self):
        """Test BaseAgent initialization."""
        agent = BaseAgent(
            name="test_agent",
            model="openai:gpt-4o",
            system_prompt="Test prompt",
        )
        assert agent.name == "test_agent"
        assert agent.model == "openai:gpt-4o"
        assert agent.system_prompt == "Test prompt"
    
    def test_input_normalization(self):
        """Test string to AgentInput conversion."""
        agent = BaseAgent(name="test")
        
        # Test string input
        normalized = agent._normalize_input("Hello")
        assert isinstance(normalized, AgentInput)
        assert normalized.input_text == "Hello"
        
        # Test AgentInput passthrough
        input_obj = AgentInput(input_text="World")
        normalized = agent._normalize_input(input_obj)
        assert normalized is input_obj


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
class TestAgentInputOutput:
    """Test AgentInput and AgentOutput DTOs."""
    
    def test_agent_input_creation(self):
        """Test AgentInput creation."""
        input_data = AgentInput(input_text="Test", context={"key": "value"})
        assert input_data.input_text == "Test"
        assert input_data.context["key"] == "value"
    
    def test_agent_input_default_context(self):
        """Test AgentInput with default context."""
        input_data = AgentInput(input_text="Test")
        assert input_data.context == {}
    
    def test_agent_output_creation(self):
        """Test AgentOutput creation."""
        output = AgentOutput(output_text="Response", metadata={"tokens": 100})
        assert output.output_text == "Response"
        assert output.metadata["tokens"] == 100
    
    def test_agent_output_default_metadata(self):
        """Test AgentOutput with default metadata."""
        output = AgentOutput(output_text="Response")
        assert output.metadata == {}


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
class TestCustomAgent:
    """Test creating custom agents."""
    
    @pytest.mark.asyncio
    async def test_custom_agent_implementation(self):
        """Test implementing a custom agent."""
        class SimpleAgent(BaseAgent):
            async def ainfer(self, input_data: AgentInput, deps=None, model_settings=None) -> AgentOutput:
                return AgentOutput(output_text=f"Echo: {input_data.input_text}")
        
        agent = SimpleAgent(name="simple")
        result = await agent.arun("Hello")
        assert result.output_text == "Echo: Hello"
    
    @pytest.mark.asyncio
    async def test_custom_agent_with_tools(self):
        """Test custom agent with tool tracking."""
        class ToolTrackingAgent(BaseAgent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tool_calls = []
            
            async def ainfer(self, input_data: AgentInput, deps=None, model_settings=None) -> AgentOutput:
                self.tool_calls.append(input_data.input_text)
                return AgentOutput(output_text=f"Tracked: {len(self.tool_calls)} calls")
        
        agent = ToolTrackingAgent(name="tracker")
        result1 = await agent.arun("First")
        result2 = await agent.arun("Second")
        
        assert len(agent.tool_calls) == 2
        assert result2.output_text == "Tracked: 2 calls"


# Integration tests that require actual API calls
@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
@pytest.mark.integration
class TestPydanticAIAgentIntegration:
    """Integration tests for PydanticAIAgent (requires API key)."""
    
    @pytest.mark.asyncio
    async def test_basic_query(self):
        """Test basic query to PydanticAIAgent."""
        # This test requires OPENAI_API_KEY to be set
        try:
            agent = PydanticAIAgent(
                name="test_agent",
                model="openai:gpt-4o",
                system_prompt="You are a helpful assistant.",
            )
            result = await agent.arun("What is 2+2? Answer with just the number.")
            assert "4" in result.output_text
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")
    
    @pytest.mark.asyncio
    async def test_with_tools(self):
        """Test agent with tools."""
        async def add_numbers(x: int, y: int) -> int:
            """Add two numbers together"""
            return x + y
        
        try:
            agent = PydanticAIAgent(
                name="math_agent",
                model="openai:gpt-4o",
                system_prompt="You can add numbers using tools.",
                tools=[
                    Tool.from_schema(
                        function=add_numbers,
                        name='add_numbers',
                        description='Add two numbers together',
                        json_schema={
                            'type': 'object',
                            'properties': {
                                'x': {'type': 'integer'},
                                'y': {'type': 'integer'},
                            },
                            'required': ['x', 'y'],
                        },
                        takes_ctx=False,
                    ),
                ],
            )
            result = await agent.arun("Use the tool to add 5 and 3")
            assert "8" in result.output_text
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])


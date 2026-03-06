"""
Tests for LLM Gateway Tools.

These tests verify that the LLM gateway tools function correctly
and can be registered with the tool system.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from ai_agents.components.tools.llm_gateway_tools import (
    llm_generate,
    llm_generate_structured,
    llm_count_tokens,
    llm_multi_model_generate,
    llm_summarize,
    llm_extract_entities,
    llm_translate,
    get_all_llm_gateway_tools,
)


# === Mock Gateway for Testing ===

class MockGateway:
    """Mock LLM Gateway for testing."""
    
    def __init__(self, model="openai:gpt-4o", **kwargs):
        self.model = model
        self.default_temperature = kwargs.get("default_temperature", 0.7)
    
    def generate(self, request):
        """Mock generate method."""
        mock_response = Mock()
        mock_response.content = f"Mock response to: {request.user_prompt}"
        mock_response.model = self.model
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_response.finish_reason = "stop"
        mock_response.metadata = {}
        return mock_response
    
    def count_tokens(self, text, model=None):
        """Mock count_tokens method."""
        # Simple mock: count words as tokens
        return len(text.split())


# === Test Tool Functions ===

@pytest.mark.asyncio
async def test_llm_generate():
    """Test basic LLM generation."""
    with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=MockGateway()):
        result = await llm_generate(
            user_prompt="Test prompt",
            system_prompt="Test system",
            temperature=0.5,
        )
        
        assert "content" in result
        assert "model" in result
        assert "usage" in result
        assert "Mock response" in result["content"]


@pytest.mark.asyncio
async def test_llm_generate_structured():
    """Test structured output generation."""
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "year": {"type": "integer"}
        }
    }
    
    # Mock response that returns valid JSON
    mock_gateway = MockGateway()
    mock_response = Mock()
    mock_response.content = '{"title": "Test", "year": 2023}'
    mock_response.model = "openai:gpt-4o"
    mock_response.usage = {"total_tokens": 30}
    mock_gateway.generate = Mock(return_value=mock_response)
    
    with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=mock_gateway):
        result = await llm_generate_structured(
            user_prompt="Test prompt",
            schema=schema,
        )
        
        assert "structured_output" in result
        assert "raw_content" in result
        assert result["structured_output"]["title"] == "Test"
        assert result["structured_output"]["year"] == 2023


@pytest.mark.asyncio
async def test_llm_count_tokens():
    """Test token counting."""
    with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=MockGateway()):
        result = await llm_count_tokens(
            text="This is a test sentence with several words",
        )
        
        assert "token_count" in result
        assert "model" in result
        assert "text_length" in result
        assert result["token_count"] > 0


@pytest.mark.asyncio
async def test_llm_multi_model_generate():
    """Test multi-model generation."""
    with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=MockGateway()):
        result = await llm_multi_model_generate(
            user_prompt="Test prompt",
            models=["openai:gpt-4o", "mistral:mistral-large-latest"],
        )
        
        assert "responses" in result
        assert "model_count" in result
        assert result["model_count"] == 2
        assert len(result["responses"]) == 2


@pytest.mark.asyncio
async def test_llm_summarize():
    """Test text summarization."""
    with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=MockGateway()):
        result = await llm_summarize(
            text="This is a long text that needs to be summarized into a shorter version.",
            max_length=10,
        )
        
        assert "summary" in result
        assert "original_length" in result
        assert "summary_length" in result
        assert "model" in result
        assert result["original_length"] > 0


@pytest.mark.asyncio
async def test_llm_extract_entities():
    """Test entity extraction."""
    # Mock response with valid JSON entities
    mock_gateway = MockGateway()
    mock_response = Mock()
    mock_response.content = '{"person": ["John Doe"], "organization": ["Acme Corp"]}'
    mock_response.model = "openai:gpt-4o"
    mock_response.usage = {"total_tokens": 30}
    mock_gateway.generate = Mock(return_value=mock_response)
    
    with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=mock_gateway):
        result = await llm_extract_entities(
            text="John Doe works at Acme Corp.",
            entity_types=["person", "organization"],
        )
        
        assert "entities" in result
        assert "model" in result
        assert "person" in result["entities"]
        assert "organization" in result["entities"]


@pytest.mark.asyncio
async def test_llm_translate():
    """Test translation."""
    with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=MockGateway()):
        result = await llm_translate(
            text="Hello",
            target_language="Spanish",
        )
        
        assert "translation" in result
        assert "source_language" in result
        assert "target_language" in result
        assert "model" in result
        assert result["target_language"] == "Spanish"


# === Test Tool Configuration ===

def test_get_all_llm_gateway_tools():
    """Test getting all tool configurations."""
    tools = get_all_llm_gateway_tools()
    
    assert isinstance(tools, list)
    assert len(tools) > 0
    
    # Check structure of first tool
    tool = tools[0]
    assert "name" in tool
    assert "function" in tool
    assert "description" in tool
    assert "json_schema" in tool
    assert "category" in tool
    assert "tags" in tool
    
    # Check that all expected tools are present
    tool_names = [t["name"] for t in tools]
    expected_tools = [
        "llm_generate",
        "llm_generate_structured",
        "llm_count_tokens",
        "llm_multi_model_generate",
        "llm_summarize",
        "llm_extract_entities",
        "llm_translate",
    ]
    
    for expected in expected_tools:
        assert expected in tool_names, f"Tool '{expected}' not found in configuration"


def test_tool_schemas_valid():
    """Test that all tool schemas are valid JSON schemas."""
    tools = get_all_llm_gateway_tools()
    
    for tool in tools:
        schema = tool["json_schema"]
        
        # Basic schema validation
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert isinstance(schema["properties"], dict)
        assert isinstance(schema["required"], list)
        
        # Check that required fields exist in properties
        for required_field in schema["required"]:
            assert required_field in schema["properties"], \
                f"Required field '{required_field}' not in properties for tool '{tool['name']}'"


def test_tool_functions_are_async():
    """Test that all tool functions are async."""
    tools = get_all_llm_gateway_tools()
    
    for tool in tools:
        func = tool["function"]
        assert asyncio.iscoroutinefunction(func), \
            f"Tool function '{tool['name']}' is not async"


def test_tool_categories_and_tags():
    """Test that all tools have proper categories and tags."""
    tools = get_all_llm_gateway_tools()
    
    for tool in tools:
        assert "category" in tool
        assert tool["category"] == "llm", f"Tool '{tool['name']}' should have 'llm' category"
        
        assert "tags" in tool
        assert isinstance(tool["tags"], list)
        assert len(tool["tags"]) > 0, f"Tool '{tool['name']}' should have at least one tag"
        assert "llm" in tool["tags"], f"Tool '{tool['name']}' should have 'llm' tag"


# === Integration Tests ===

@pytest.mark.asyncio
async def test_tool_with_registry():
    """Test registering tools with ToolRegistry."""
    try:
        from ai_core.interfaces.tool_registry import ToolRegistry
        
        registry = ToolRegistry()
        tools = get_all_llm_gateway_tools()
        
        # Register all tools
        for tool_config in tools:
            registry.register_direct_tool(
                name=tool_config["name"],
                function=tool_config["function"],
                description=tool_config["description"],
                json_schema=tool_config["json_schema"],
                category=tool_config.get("category"),
                tags=tool_config.get("tags", []),
            )
        
        # Verify tools are registered
        all_tools = registry.get_all_tools()
        assert len(all_tools) == len(tools)
        
        # Verify we can get tools by category
        llm_tools = registry.get_tools_by_category("llm")
        assert len(llm_tools) == len(tools)
        
    except ImportError:
        pytest.skip("ToolRegistry not available")


@pytest.mark.asyncio
async def test_tool_invocation_through_invoker():
    """Test invoking tools through ToolInvoker."""
    try:
        from ai_core.interfaces.tool_registry import ToolRegistry
        from ai_core.interfaces.tool_invoker import ToolInvoker
        
        # Create registry and register tools
        registry = ToolRegistry()
        tools = get_all_llm_gateway_tools()
        
        for tool_config in tools:
            registry.register_direct_tool(
                name=tool_config["name"],
                function=tool_config["function"],
                description=tool_config["description"],
                json_schema=tool_config["json_schema"],
            )
        
        # Create invoker
        invoker = ToolInvoker(registry)
        
        # Test invoking a tool
        with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=MockGateway()):
            result = await invoker.invoke_tool(
                "llm_generate",
                {"user_prompt": "Test"}
            )
            
            assert result is not None
            assert "content" in result
        
    except ImportError:
        pytest.skip("ToolInvoker not available")


# === Error Handling Tests ===

@pytest.mark.asyncio
async def test_llm_generate_handles_errors():
    """Test that llm_generate handles errors gracefully."""
    mock_gateway = MockGateway()
    mock_gateway.generate = Mock(side_effect=Exception("API Error"))
    
    with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=mock_gateway):
        with pytest.raises(Exception):
            await llm_generate(user_prompt="Test")


@pytest.mark.asyncio
async def test_llm_generate_structured_handles_invalid_json():
    """Test that structured generation handles invalid JSON."""
    mock_gateway = MockGateway()
    mock_response = Mock()
    mock_response.content = "Not valid JSON"
    mock_response.model = "openai:gpt-4o"
    mock_response.usage = {"total_tokens": 30}
    mock_gateway.generate = Mock(return_value=mock_response)
    
    with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=mock_gateway):
        result = await llm_generate_structured(
            user_prompt="Test",
            schema={"type": "object"}
        )
        
        # Should return error in structured_output
        assert "error" in result["structured_output"]


# === Performance Tests ===

@pytest.mark.asyncio
async def test_multi_model_runs_in_parallel():
    """Test that multi-model generation runs models in parallel."""
    import time
    
    # Create mock gateway with delay
    async def mock_generate_with_delay(*args, **kwargs):
        await asyncio.sleep(0.1)  # Simulate API call
        return MockGateway().generate(Mock(user_prompt="test"))
    
    mock_gateway = MockGateway()
    mock_gateway.generate = mock_generate_with_delay
    
    with patch('ai_agents.components.tools.llm_gateway_tools.get_llm_gateway', return_value=mock_gateway):
        start = time.time()
        result = await llm_multi_model_generate(
            user_prompt="Test",
            models=["model1", "model2", "model3"]
        )
        elapsed = time.time() - start
        
        # Should take ~0.1s (parallel) not ~0.3s (sequential)
        assert elapsed < 0.2, "Multi-model generation should run in parallel"
        assert result["model_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


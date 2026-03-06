"""
Example demonstrating agents using the ToolRegistry system.

This shows how to:
1. Create a ToolRegistry and populate it with tools
2. Use the registry with BaseAgent
3. Load tools from configuration files
4. Mix direct and MCP tools
"""
import asyncio
from pathlib import Path
import dotenv

dotenv.load_dotenv()

from ai_core.interfaces import (
    ToolRegistry,
    ToolInvoker,
    ToolLoader,
    ModelSettings,
)
from ai_agents.components import BaseAgent
from ai_agents.components.tools import (
    get_current_weather,
    calculate_distance,
    format_currency,
    analyze_sentiment,
    WebSearchTool,
)


async def example_basic_agent_with_registry():
    """Example 1: Basic agent with manually registered tools."""
    print("\n" + "="*60)
    print("Example 1: Agent with Manual Tool Registration")
    print("="*60)
    
    # Create registry
    registry = ToolRegistry()
    
    # Register some tools
    registry.register_direct_tool(
        name="get_weather",
        function=get_current_weather,
        description="Get the current weather for a location",
        json_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        category="utility",
        tags=["weather", "external"]
    )
    
    registry.register_direct_tool(
        name="format_currency",
        function=format_currency,
        description="Format a number as currency",
        json_schema={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to format"},
                "currency": {
                    "type": "string",
                    "enum": ["USD", "EUR", "GBP", "JPY"],
                    "default": "USD"
                }
            },
            "required": ["amount"],
            "additionalProperties": False,
        },
        category="utility",
        tags=["formatting"]
    )
    
    registry.register_direct_tool(
        name="analyze_sentiment",
        function=analyze_sentiment,
        description="Analyze the sentiment of text",
        json_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"}
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        category="nlp",
        tags=["sentiment", "analysis"]
    )
    
    print(f"✓ Registered {len(registry)} tools in registry")
    
    # Create agent with registry
    agent = BaseAgent(
        name="registry_agent",
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant with access to various tools.",
        tool_registry=registry,
    )
    
    print("✓ Created agent with tool registry")
    
    # Test the agent
    queries = [
        "What's the weather like in Paris?",
        "Format 12345.67 as US dollars",
        "Analyze the sentiment: 'I love this amazing new feature!'",
    ]
    
    for query in queries:
        print(f"\n→ Query: {query}")
        result = await agent.arun(
            query,
            model_settings=ModelSettings(parallel_tool_calls=True)
        )
        print(f"← Response: {result.output_text}")


async def example_agent_with_config_loader():
    """Example 2: Agent with tools loaded from configuration."""
    print("\n" + "="*60)
    print("Example 2: Agent with Config-Loaded Tools")
    print("="*60)
    
    # Create a config file
    config_content = """
tools:
  - name: calculate_distance
    type: direct
    module: ai_agents.components.tools.example_tools
    function: calculate_distance
    description: Calculate distance between two geographic coordinates
    category: math
    tags: [geography, calculation]
    schema:
      type: object
      properties:
        lat1:
          type: number
          description: Latitude of first point
        lon1:
          type: number
          description: Longitude of first point
        lat2:
          type: number
          description: Latitude of second point
        lon2:
          type: number
          description: Longitude of second point
      required: [lat1, lon1, lat2, lon2]
      additionalProperties: false
  
  - name: get_weather
    type: direct
    module: ai_agents.components.tools.example_tools
    function: get_current_weather
    description: Get current weather for a location
    category: utility
    tags: [weather]
    schema:
      type: object
      properties:
        location:
          type: string
          description: City name
        units:
          type: string
          enum: [celsius, fahrenheit]
          default: celsius
      required: [location]
      additionalProperties: false
"""
    
    # Save config
    config_path = Path("temp_agent_tools.yaml")
    config_path.write_text(config_content)
    
    try:
        # Create registry and load tools
        registry = ToolRegistry()
        loader = ToolLoader(registry)
        count = loader.load_from_yaml(str(config_path))
        
        print(f"✓ Loaded {count} tools from configuration")
        
        # Create agent
        agent = BaseAgent(
            name="config_agent",
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tool_registry=registry,
        )
        
        print("✓ Created agent with config-loaded tools")
        
        # Test
        query = "What's the weather in London and calculate the distance from New York (40.7128, -74.0060) to London (51.5074, -0.1278)"
        print(f"\n→ Query: {query}")
        
        result = await agent.arun(
            query,
            model_settings=ModelSettings(parallel_tool_calls=True)
        )
        print(f"← Response: {result.output_text}")
        
    finally:
        if config_path.exists():
            config_path.unlink()


async def example_agent_with_tool_categories():
    """Example 3: Using tool categories to organize tools."""
    print("\n" + "="*60)
    print("Example 3: Organizing Tools by Category")
    print("="*60)
    
    # Create registry
    registry = ToolRegistry()
    
    # Register tools in different categories
    tools_config = [
        # Math tools
        {
            "name": "calculate_distance",
            "function": calculate_distance,
            "description": "Calculate distance between coordinates",
            "category": "math",
            "tags": ["geography"],
            "json_schema": {
                "type": "object",
                "properties": {
                    "lat1": {"type": "number"},
                    "lon1": {"type": "number"},
                    "lat2": {"type": "number"},
                    "lon2": {"type": "number"},
                },
                "required": ["lat1", "lon1", "lat2", "lon2"],
                "additionalProperties": False,
            }
        },
        # Utility tools
        {
            "name": "get_weather",
            "function": get_current_weather,
            "description": "Get current weather",
            "category": "utility",
            "tags": ["weather"],
            "json_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
                "additionalProperties": False,
            }
        },
        {
            "name": "format_currency",
            "function": format_currency,
            "description": "Format as currency",
            "category": "utility",
            "tags": ["formatting"],
            "json_schema": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY"]},
                },
                "required": ["amount"],
                "additionalProperties": False,
            }
        },
        # NLP tools
        {
            "name": "analyze_sentiment",
            "function": analyze_sentiment,
            "description": "Analyze text sentiment",
            "category": "nlp",
            "tags": ["analysis"],
            "json_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            }
        },
    ]
    
    for tool_config in tools_config:
        registry.register_direct_tool(**tool_config)
    
    # Show organization
    print(f"✓ Registered {len(registry)} tools")
    print("\nTools by category:")
    for category in registry.list_categories():
        tools = registry.get_tools_by_category(category)
        print(f"  {category}: {', '.join(t.name for t in tools)}")
    
    # Create specialized agents for different categories
    print("\n→ Creating specialized agents...")
    
    # Math agent - only math tools
    math_registry = ToolRegistry()
    for metadata in registry.get_tools_by_category("math"):
        math_registry.register_tool_instance(metadata.tool_instance)
    
    # Utility agent - only utility tools
    utility_registry = ToolRegistry()
    for metadata in registry.get_tools_by_category("utility"):
        utility_registry.register_tool_instance(metadata.tool_instance)
    
    print(f"  Math agent: {len(math_registry)} tools")
    print(f"  Utility agent: {len(utility_registry)} tools")
    
    # Test utility agent
    utility_agent = BaseAgent(
        name="utility_agent",
        model="openai:gpt-4o-mini",
        system_prompt="You are a utility assistant.",
        tool_registry=utility_registry,
    )
    
    query = "What's the weather in Tokyo and format 1000 as Yen?"
    print(f"\n→ Utility Agent Query: {query}")
    result = await utility_agent.arun(query)
    print(f"← Response: {result.output_text}")


async def example_direct_tool_invocation():
    """Example 4: Direct tool invocation using ToolInvoker."""
    print("\n" + "="*60)
    print("Example 4: Direct Tool Invocation (No Agent)")
    print("="*60)
    
    # Create registry
    registry = ToolRegistry()
    registry.register_direct_tool(
        name="get_weather",
        function=get_current_weather,
        description="Get weather",
        json_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string"},
            },
            "required": ["location"],
            "additionalProperties": False,
        }
    )
    
    # Create invoker
    invoker = ToolInvoker(registry)
    
    # Invoke tool directly
    print("\n→ Invoking tool: get_weather(location='Berlin')")
    result = await invoker.ainvoke("get_weather", location="Berlin")
    print(f"← Result: {result}")
    
    # Show tool info
    print("\n→ Tool information:")
    info = invoker.get_tool_info("get_weather")
    for key, value in info.items():
        print(f"  {key}: {value}")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("AGENT WITH TOOL REGISTRY - COMPREHENSIVE EXAMPLES")
    print("="*60)
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set")
        print("   Some examples will be skipped")
        print("   Set the environment variable to run all examples\n")
    
    try:
        # Run examples that don't require API key
        await example_direct_tool_invocation()
        
        # Run agent examples if API key is available
        if os.getenv("OPENAI_API_KEY"):
            await example_basic_agent_with_registry()
            await example_agent_with_config_loader()
            await example_agent_with_tool_categories()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


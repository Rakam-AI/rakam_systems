"""
Comprehensive example demonstrating the uniform tool invocation system.

This example shows:
1. Creating and registering tools (both direct and MCP modes)
2. Using ToolRegistry for tool management
3. Using ToolInvoker for uniform invocation
4. Loading tools from configuration files
5. Using tools with agents
6. Both synchronous and asynchronous invocation
"""
import asyncio
from datetime import datetime
from pathlib import Path
import dotenv

dotenv.load_dotenv()

# Import framework components
from ai_core.interfaces import (
    ToolRegistry,
    ToolInvoker,
    ToolLoader,
    Tool,
    ModelSettings,
)
from ai_agents.components import BaseAgent
from ai_agents.components.tools import (
    get_current_weather,
    calculate_distance,
    format_currency,
    translate_text,
    analyze_sentiment,
    WebSearchTool,
    DatabaseQueryTool,
    get_all_example_tools,
)
from ai_core.mcp.mcp_server import MCPServer


# === Part 1: Manual Tool Registration ===

def example_manual_registration():
    """Example of manually registering tools."""
    print("\n" + "="*60)
    print("PART 1: Manual Tool Registration")
    print("="*60)
    
    # Create registry
    registry = ToolRegistry()
    
    # Register direct tools using helper function
    for tool_config in get_all_example_tools():
        registry.register_direct_tool(
            name=tool_config["name"],
            function=tool_config["function"],
            description=tool_config["description"],
            json_schema=tool_config["json_schema"],
            category=tool_config.get("category"),
            tags=tool_config.get("tags", []),
        )
    
    # Register tool components
    registry.register_tool_instance(
        WebSearchTool(),
        category="search",
        tags=["web", "external"]
    )
    
    registry.register_tool_instance(
        DatabaseQueryTool(),
        category="database",
        tags=["data", "query"]
    )
    
    print(f"\n✓ Registered {len(registry)} tools")
    print(f"  Categories: {', '.join(registry.list_categories())}")
    print(f"  Tags: {', '.join(registry.list_tags())}")
    
    # List tools by category
    print("\nTools by category:")
    for category in registry.list_categories():
        tools = registry.get_tools_by_category(category)
        print(f"  {category}: {', '.join(t.name for t in tools)}")
    
    return registry


# === Part 2: Tool Invocation ===

async def example_tool_invocation(registry: ToolRegistry):
    """Example of invoking tools using ToolInvoker."""
    print("\n" + "="*60)
    print("PART 2: Tool Invocation (Direct Mode)")
    print("="*60)
    
    # Create invoker
    invoker = ToolInvoker(registry)
    
    # Example 1: Get weather
    print("\n[Example 1] Get current weather:")
    result = await invoker.ainvoke("get_current_weather", location="San Francisco")
    print(f"  Weather in {result['location']}: {result['temperature']}°{result['units'][0].upper()}, {result['condition']}")
    
    # Example 2: Calculate distance
    print("\n[Example 2] Calculate distance:")
    distance = await invoker.ainvoke(
        "calculate_distance",
        lat1=40.7128, lon1=-74.0060,  # New York
        lat2=34.0522, lon2=-118.2437   # Los Angeles
    )
    print(f"  Distance: {distance} km")
    
    # Example 3: Format currency
    print("\n[Example 3] Format currency:")
    formatted = await invoker.ainvoke("format_currency", amount=1234.56, currency="EUR")
    print(f"  Formatted: {formatted}")
    
    # Example 4: Analyze sentiment
    print("\n[Example 4] Analyze sentiment:")
    sentiment = await invoker.ainvoke(
        "analyze_sentiment",
        text="This is a wonderful day! I feel great and everything is amazing!"
    )
    print(f"  Sentiment: {sentiment['sentiment']} (score: {sentiment['score']})")
    
    # Example 5: Web search (using ToolComponent)
    print("\n[Example 5] Web search:")
    search_results = await invoker.ainvoke("web_search", query="Python programming")
    print(f"  Found {search_results['total_results']} results for '{search_results['query']}'")
    for i, result in enumerate(search_results['results'], 1):
        print(f"    {i}. {result['title']}")
    
    # Example 6: Database query (using ToolComponent)
    print("\n[Example 6] Database query:")
    db_results = await invoker.ainvoke("database_query", query="SELECT * FROM users")
    print(f"  Query returned {db_results['count']} results from '{db_results['table']}' table")
    for user in db_results['results']:
        print(f"    - {user['name']} ({user['email']})")
    
    return invoker


# === Part 3: MCP Tool Registration and Invocation ===

async def example_mcp_tools(registry: ToolRegistry):
    """Example of MCP-based tool invocation."""
    print("\n" + "="*60)
    print("PART 3: MCP Tool Registration and Invocation")
    print("="*60)
    
    # Create and setup MCP server
    mcp_server = MCPServer(name="example_mcp_server")
    mcp_server.setup()
    
    # Register tool components with the MCP server
    search_tool = WebSearchTool(name="mcp_search")
    db_tool = DatabaseQueryTool(name="mcp_database")
    
    mcp_server.register_component(search_tool)
    mcp_server.register_component(db_tool)
    
    print(f"\n✓ Created MCP server with components: {mcp_server.run()}")
    
    # Register MCP tools in the registry
    registry.register_mcp_tool(
        name="mcp_web_search",
        mcp_server="example_mcp_server",
        mcp_tool_name="mcp_search",
        description="Search the web via MCP server",
        category="search",
        tags=["mcp", "web"],
    )
    
    registry.register_mcp_tool(
        name="mcp_database_query",
        mcp_server="example_mcp_server",
        mcp_tool_name="mcp_database",
        description="Query database via MCP server",
        category="database",
        tags=["mcp", "data"],
    )
    
    print(f"✓ Registered {len(registry.get_tools_by_mode('mcp'))} MCP tools")
    
    # Create invoker and register MCP server
    invoker = ToolInvoker(registry)
    invoker.register_mcp_server("example_mcp_server", mcp_server)
    
    # Invoke MCP tools
    print("\n[Example 1] MCP Web Search:")
    result = await invoker.ainvoke("mcp_web_search", query="AI agents")
    print(f"  Found {result['total_results']} results via MCP")
    
    print("\n[Example 2] MCP Database Query:")
    result = await invoker.ainvoke("mcp_database_query", query="SELECT * FROM products")
    print(f"  Retrieved {result['count']} products via MCP")
    
    return invoker, mcp_server


# === Part 4: Loading Tools from Configuration ===

def example_config_loading(registry: ToolRegistry):
    """Example of loading tools from configuration file."""
    print("\n" + "="*60)
    print("PART 4: Loading Tools from Configuration")
    print("="*60)
    
    # Create a simple config file for demonstration
    config_content = """
tools:
  - name: get_timestamp
    type: direct
    module: datetime
    function: now
    description: Get current timestamp
    category: utility
    tags: [time, datetime]
    schema:
      type: object
      properties: {}
      additionalProperties: false
"""
    
    # Save to temporary file
    config_path = Path("temp_tools_config.yaml")
    config_path.write_text(config_content)
    
    try:
        # Load tools from config
        loader = ToolLoader(registry)
        count = loader.load_from_yaml(str(config_path))
        
        print(f"\n✓ Loaded {count} tool(s) from configuration")
        
        # Show the loaded tool
        tool = registry.get_tool("get_timestamp")
        if tool:
            print(f"  Tool: {tool.name}")
            print(f"  Description: {tool.description}")
            print(f"  Category: {tool.category}")
            print(f"  Mode: {tool.mode}")
    finally:
        # Clean up
        if config_path.exists():
            config_path.unlink()


# === Part 5: Using Tools with Agents ===

async def example_agent_with_tools():
    """Example of using tools with BaseAgent."""
    print("\n" + "="*60)
    print("PART 5: Using Tools with Agents")
    print("="*60)
    
    # Create tools for the agent
    tools = [
        Tool.from_schema(
            function=get_current_weather,
            name="get_current_weather",
            description="Get the current weather for a location",
            json_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        ),
        Tool.from_schema(
            function=format_currency,
            name="format_currency",
            description="Format a number as currency",
            json_schema={
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to format"},
                    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY"], "default": "USD"},
                },
                "required": ["amount"],
                "additionalProperties": False,
            },
        ),
        Tool.from_schema(
            function=analyze_sentiment,
            name="analyze_sentiment",
            description="Analyze sentiment of text",
            json_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
    ]
    
    # Create agent with tools
    agent = BaseAgent(
        name="tool_agent",
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant with access to various tools. Use them to help answer questions.",
        tools=tools,
    )
    
    print("\n✓ Created agent with tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Test 1: Weather query
    print("\n[Test 1] Ask about weather:")
    result = await agent.arun(
        "What's the weather like in Tokyo?",
        model_settings=ModelSettings(parallel_tool_calls=True)
    )
    print(f"  Agent: {result.output_text}")
    
    # Test 2: Currency formatting
    print("\n[Test 2] Format currency:")
    result = await agent.arun(
        "Format 9999.99 in Japanese Yen",
        model_settings=ModelSettings(parallel_tool_calls=True)
    )
    print(f"  Agent: {result.output_text}")
    
    # Test 3: Sentiment analysis
    print("\n[Test 3] Analyze sentiment:")
    result = await agent.arun(
        "What's the sentiment of this text: 'I'm having a terrible day and nothing is going right.'",
        model_settings=ModelSettings(parallel_tool_calls=True)
    )
    print(f"  Agent: {result.output_text}")


# === Part 6: Tool Information and Discovery ===

def example_tool_discovery(registry: ToolRegistry, invoker: ToolInvoker):
    """Example of discovering and inspecting tools."""
    print("\n" + "="*60)
    print("PART 6: Tool Discovery and Information")
    print("="*60)
    
    # List all available tools
    print("\nAll available tools:")
    tools = invoker.list_available_tools()
    for name, info in tools.items():
        print(f"  - {name} ({info['mode']}): {info['description']}")
    
    # Get tools by category
    print("\nTools by category 'nlp':")
    nlp_tools = registry.get_tools_by_category("nlp")
    for tool in nlp_tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Get tools by tag
    print("\nTools with tag 'external':")
    external_tools = registry.get_tools_by_tag("external")
    for tool in external_tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Get specific tool info
    print("\nDetailed info for 'get_current_weather':")
    info = invoker.get_tool_info("get_current_weather")
    if info:
        for key, value in info.items():
            print(f"  {key}: {value}")


# === Main Example Runner ===

async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("UNIFORM TOOL INVOCATION SYSTEM - COMPREHENSIVE EXAMPLE")
    print("="*60)
    
    try:
        # Part 1: Manual registration
        registry = example_manual_registration()
        
        # Part 2: Tool invocation
        invoker = await example_tool_invocation(registry)
        
        # Part 3: MCP tools
        await example_mcp_tools(registry)
        
        # Part 4: Config loading
        example_config_loading(registry)
        
        # Part 5: Agent integration (requires API key)
        import os
        if os.getenv("OPENAI_API_KEY"):
            await example_agent_with_tools()
        else:
            print("\n" + "="*60)
            print("PART 5: Skipped (no OPENAI_API_KEY)")
            print("="*60)
            print("\nSet OPENAI_API_KEY environment variable to test agent integration")
        
        # Part 6: Tool discovery
        example_tool_discovery(registry, invoker)
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Summary
        print("\nSummary:")
        print(f"  Total tools registered: {len(registry)}")
        print(f"  Direct tools: {len(registry.get_tools_by_mode('direct'))}")
        print(f"  MCP tools: {len(registry.get_tools_by_mode('mcp'))}")
        print(f"  Categories: {len(registry.list_categories())}")
        print(f"  Tags: {len(registry.list_tags())}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


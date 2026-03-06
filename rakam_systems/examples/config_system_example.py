"""
Comprehensive example demonstrating the configuration system.

This example shows:
1. Loading agent configurations from YAML
2. Creating agents from configuration
3. Using agents with tracking enabled
4. Exporting tracking data to CSV
5. Configuration validation
"""
import asyncio
from pathlib import Path
import dotenv

dotenv.load_dotenv()

from ai_core.config_loader import ConfigurationLoader
from ai_core.interfaces import ModelSettings
from ai_core.tracking import get_tracking_manager


async def example_1_load_and_validate():
    """Example 1: Load and validate configuration."""
    print("\n" + "="*80)
    print("Example 1: Load and Validate Configuration")
    print("="*80)
    
    # Create loader
    loader = ConfigurationLoader()
    
    # Load configuration
    config_path = Path(__file__).parent / "configs" / "complete_agent_config.yaml"
    
    try:
        config = loader.load_from_yaml(str(config_path))
        print(f"✓ Successfully loaded configuration")
        print(f"  - Version: {config.version}")
        print(f"  - Prompts: {len(config.prompts)}")
        print(f"  - Tools: {len(config.tools)}")
        print(f"  - Agents: {len(config.agents)}")
        
        # List available components
        print("\n📝 Available Prompts:")
        for name, prompt in config.prompts.items():
            print(f"  - {name}: {prompt.description}")
        
        print("\n🔧 Available Tools:")
        for name, tool in config.tools.items():
            print(f"  - {name} ({tool.type}): {tool.description}")
        
        print("\n🤖 Available Agents:")
        for name, agent in config.agents.items():
            print(f"  - {name}: {agent.description}")
            print(f"    Model: {agent.llm_config.model}")
            print(f"    Tools: {len(agent.tools)}")
            print(f"    Tracking: {agent.enable_tracking}")
        
        # Validate configuration
        print("\n🔍 Validating configuration...")
        is_valid, errors = loader.validate_config()
        
        if is_valid:
            print("✓ Configuration is valid!")
        else:
            print("✗ Configuration has errors:")
            for error in errors:
                print(f"  - {error}")
        
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        import traceback
        traceback.print_exc()


async def example_2_create_agent_from_config():
    """Example 2: Create and use an agent from configuration."""
    print("\n" + "="*80)
    print("Example 2: Create Agent from Configuration")
    print("="*80)
    
    loader = ConfigurationLoader()
    config_path = Path(__file__).parent / "configs" / "complete_agent_config.yaml"
    
    # Load configuration
    loader.load_from_yaml(str(config_path))
    
    # Create a specific agent
    print("\n→ Creating 'general_agent' from configuration...")
    agent = loader.create_agent("general_agent")
    
    print(f"✓ Created agent: {agent.name}")
    print(f"  Model: {agent.model}")
    print(f"  System prompt: {agent.system_prompt[:100]}...")
    print(f"  Tools: {len(agent.tools)}")
    print(f"  Tracking enabled: {agent._tracking_enabled}")
    
    # Use the agent
    print("\n→ Testing agent...")
    query = "What's the weather like in Paris?"
    print(f"  Query: {query}")
    
    try:
        result = await agent.arun(query)
        print(f"  Response: {result.output_text}")
    except Exception as e:
        print(f"  Note: {e} (This is expected if tools aren't fully implemented)")


async def example_3_tracking_and_export():
    """Example 3: Use agent with tracking and export data."""
    print("\n" + "="*80)
    print("Example 3: Agent Tracking and Data Export")
    print("="*80)
    
    loader = ConfigurationLoader()
    config_path = Path(__file__).parent / "configs" / "complete_agent_config.yaml"
    
    # Load and create agent with tracking
    loader.load_from_yaml(str(config_path))
    agent = loader.create_agent("fast_agent")  # Using fast_agent (no tools)
    
    print(f"✓ Created agent with tracking: {agent.name}")
    print(f"  Tracking directory: {agent._tracking_output_dir}")
    
    # Start a tracking session
    manager = agent.get_tracking_manager()
    session_id = manager.start_session(agent.name)
    print(f"  Started tracking session: {session_id}")
    
    # Make several calls
    queries = [
        "What is 2 + 2?",
        "Explain Python in one sentence.",
        "What is the capital of France?",
    ]
    
    print("\n→ Making tracked calls...")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")
        try:
            result = await agent.arun(query)
            print(f"     Response: {result.output_text[:100]}...")
        except Exception as e:
            print(f"     Error: {e}")
    
    # End session
    manager.end_session(session_id)
    print("\n✓ Session completed")
    
    # Get statistics
    stats = agent.get_tracking_statistics()
    print("\n📊 Tracking Statistics:")
    print(f"  Total calls: {stats.get('total_calls', 0)}")
    print(f"  Successful: {stats.get('successful_calls', 0)}")
    print(f"  Failed: {stats.get('failed_calls', 0)}")
    print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
    print(f"  Avg duration: {stats.get('average_duration_seconds', 0):.2f}s")
    
    # Export to CSV
    print("\n→ Exporting tracking data...")
    csv_path = agent.export_tracking_data(format='csv')
    print(f"✓ Exported to CSV: {csv_path}")
    
    # Export to JSON
    json_path = agent.export_tracking_data(format='json')
    print(f"✓ Exported to JSON: {json_path}")
    
    # Show CSV preview
    print("\n📄 CSV Preview (first few lines):")
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()[:5]  # First 5 lines
            for line in lines:
                print(f"  {line.rstrip()}")
    except Exception as e:
        print(f"  Could not read CSV: {e}")


async def example_4_create_all_agents():
    """Example 4: Create all agents from configuration."""
    print("\n" + "="*80)
    print("Example 4: Create All Agents")
    print("="*80)
    
    loader = ConfigurationLoader()
    config_path = Path(__file__).parent / "configs" / "complete_agent_config.yaml"
    
    # Load configuration
    loader.load_from_yaml(str(config_path))
    
    # Create all agents
    print("\n→ Creating all agents from configuration...")
    agents = loader.create_all_agents()
    
    print(f"✓ Created {len(agents)} agents:")
    for name, agent in agents.items():
        print(f"  - {name}")
        print(f"    Model: {agent.model}")
        print(f"    Tools: {len(agent.tools)}")
        print(f"    Tracking: {agent._tracking_enabled}")
    
    # Demonstrate using different agents
    print("\n→ Testing different agent specializations...")
    
    # Fast agent (no tools)
    if "fast_agent" in agents:
        print("\n  Fast Agent (optimized for speed):")
        try:
            result = await agents["fast_agent"].arun("Quick: what is 5 * 5?")
            print(f"    Response: {result.output_text}")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Power agent (all tools)
    if "power_agent" in agents:
        print("\n  Power Agent (all capabilities):")
        print(f"    Available tools: {len(agents['power_agent'].tools)}")


async def example_5_custom_configuration():
    """Example 5: Create configuration programmatically."""
    print("\n" + "="*80)
    print("Example 5: Programmatic Configuration")
    print("="*80)
    
    from ai_core.config_schema import (
        ConfigFileSchema,
        AgentConfigSchema,
        ModelConfigSchema,
        PromptConfigSchema,
        ToolConfigSchema,
    )
    
    # Create configuration programmatically
    config_dict = {
        "version": "1.0",
        "prompts": {
            "simple_prompt": {
                "name": "simple_prompt",
                "system_prompt": "You are a helpful assistant.",
                "skills": ["General assistance"],
                "tags": ["simple"],
            }
        },
        "tools": {},
        "agents": {
            "custom_agent": {
                "name": "custom_agent",
                "description": "Programmatically created agent",
                "model_config": {
                    "model": "openai:gpt-4o-mini",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "parallel_tool_calls": True,
                },
                "prompt_config": "simple_prompt",
                "tools": [],
                "enable_tracking": True,
                "tracking_output_dir": "./agent_tracking/custom",
                "stateful": False,
                "metadata": {
                    "created_by": "example_script",
                },
            }
        }
    }
    
    print("\n→ Creating agent from programmatic configuration...")
    loader = ConfigurationLoader()
    loader.load_from_dict(config_dict)
    
    agent = loader.create_agent("custom_agent")
    print(f"✓ Created custom agent: {agent.name}")
    print(f"  Model: {agent.model}")
    
    # Test it
    print("\n→ Testing custom agent...")
    try:
        result = await agent.arun("Hello! Can you introduce yourself?")
        print(f"  Response: {result.output_text}")
    except Exception as e:
        print(f"  Error: {e}")


async def example_6_tracking_analysis():
    """Example 6: Advanced tracking analysis."""
    print("\n" + "="*80)
    print("Example 6: Advanced Tracking Analysis")
    print("="*80)
    
    loader = ConfigurationLoader()
    config_path = Path(__file__).parent / "configs" / "complete_agent_config.yaml"
    
    loader.load_from_yaml(str(config_path))
    agent = loader.create_agent("fast_agent")
    
    print(f"✓ Using agent: {agent.name}")
    
    # Create multiple sessions
    manager = agent.get_tracking_manager()
    
    # Session 1: Simple queries
    print("\n→ Session 1: Simple calculations")
    session1 = manager.start_session(agent.name, "session_simple")
    
    for query in ["What is 2+2?", "What is 10*5?", "What is 100/4?"]:
        try:
            await agent.arun(query)
        except:
            pass
    
    manager.end_session(session1)
    
    # Session 2: Complex queries
    print("→ Session 2: Complex questions")
    session2 = manager.start_session(agent.name, "session_complex")
    
    for query in [
        "Explain quantum computing in simple terms.",
        "What are the benefits of functional programming?",
    ]:
        try:
            await agent.arun(query)
        except:
            pass
    
    manager.end_session(session2)
    
    # Analyze sessions
    print("\n📊 Session Analysis:")
    
    for session_id in [session1, session2]:
        session = manager.get_session(session_id)
        if session:
            print(f"\n  Session: {session_id}")
            print(f"    Calls: {session.total_calls}")
            print(f"    Successful: {session.successful_calls}")
            print(f"    Failed: {session.failed_calls}")
            print(f"    Duration: {session.total_duration:.2f}s")
            
            # Export session-specific data
            csv_path = manager.export_to_csv(
                filename=f"{session_id}.csv",
                session_id=session_id
            )
            print(f"    Exported: {csv_path}")


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("CONFIGURATION SYSTEM - COMPREHENSIVE EXAMPLES")
    print("="*80)
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set")
        print("   Some examples may fail or be skipped")
        print("   Set the environment variable to run all examples\n")
    
    try:
        # Run examples
        await example_1_load_and_validate()
        
        if os.getenv("OPENAI_API_KEY"):
            await example_2_create_agent_from_config()
            await example_3_tracking_and_export()
            await example_4_create_all_agents()
            await example_5_custom_configuration()
            await example_6_tracking_analysis()
        else:
            print("\n⏭️  Skipping examples that require API key")
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED!")
        print("="*80)
        print("\n💡 Key Takeaways:")
        print("  1. Configuration files allow declarative agent setup")
        print("  2. Tracking captures all inputs/outputs automatically")
        print("  3. CSV export enables easy evaluation and analysis")
        print("  4. Multiple agents can share prompts and tools")
        print("  5. Validation catches configuration errors early")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


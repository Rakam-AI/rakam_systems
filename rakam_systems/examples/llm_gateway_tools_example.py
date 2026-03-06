"""
Example demonstrating LLM Gateway Tools usage.

This example shows how to:
1. Use LLM generation tools directly
2. Register LLM gateway tools with agents
3. Enable meta-reasoning workflows
4. Use specialized LLM tools (summarization, entity extraction, translation)
5. Compare outputs across multiple models
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


async def example_basic_generation():
    """Example 1: Basic LLM generation through tools."""
    print("\n" + "="*70)
    print("Example 1: Basic LLM Generation")
    print("="*70)
    
    result = await llm_generate(
        user_prompt="What are the three laws of robotics?",
        system_prompt="You are a knowledgeable assistant specializing in science fiction.",
        temperature=0.7,
    )
    
    print(f"\nModel: {result['model']}")
    print(f"\nResponse:\n{result['content']}")
    print(f"\nTokens used: {result['usage']}")


async def example_structured_output():
    """Example 2: Generate structured output with schema."""
    print("\n" + "="*70)
    print("Example 2: Structured Output Generation")
    print("="*70)
    
    # Define a schema for book information
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "publication_year": {"type": "integer"},
            "genre": {"type": "string"},
            "summary": {"type": "string"}
        },
        "required": ["title", "author", "publication_year"]
    }
    
    result = await llm_generate_structured(
        user_prompt="Tell me about the book '1984' by George Orwell",
        schema=schema,
        system_prompt="You are a literary assistant. Return information as JSON.",
    )
    
    print(f"\nModel: {result['model']}")
    print(f"\nStructured output:")
    import json
    print(json.dumps(result['structured_output'], indent=2))


async def example_token_counting():
    """Example 3: Count tokens in text."""
    print("\n" + "="*70)
    print("Example 3: Token Counting")
    print("="*70)
    
    text = """
    The quick brown fox jumps over the lazy dog. 
    This is a sample text to demonstrate token counting.
    Token counting is useful for managing context windows and estimating costs.
    """
    
    result = await llm_count_tokens(
        text=text,
        model="openai:gpt-4o",
    )
    
    print(f"\nText length: {result['text_length']} characters")
    print(f"Token count: {result['token_count']} tokens")
    print(f"Model: {result['model']}")


async def example_multi_model_comparison():
    """Example 4: Compare outputs from multiple models."""
    print("\n" + "="*70)
    print("Example 4: Multi-Model Comparison")
    print("="*70)
    
    result = await llm_multi_model_generate(
        user_prompt="What is the meaning of life?",
        models=["openai:gpt-4o", "mistral:mistral-large-latest"],
        temperature=0.8,
    )
    
    print(f"\nCompared {result['model_count']} models:\n")
    
    for i, response in enumerate(result['responses'], 1):
        print(f"\nModel {i}: {response['model']}")
        print(f"Response: {response['content'][:200]}...")
        print(f"Tokens: {response['usage']}")


async def example_summarization():
    """Example 5: Summarize text."""
    print("\n" + "="*70)
    print("Example 5: Text Summarization")
    print("="*70)
    
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals. The term "artificial intelligence" 
    had previously been used to describe machines that mimic and display "human" cognitive 
    skills that are associated with the human mind, such as "learning" and "problem-solving". 
    This definition has since been rejected by major AI researchers who now describe AI in 
    terms of rationality and acting rationally, which does not limit how intelligence can be 
    articulated. AI applications include advanced web search engines, recommendation systems, 
    understanding human speech, self-driving cars, generative or creative tools, automated 
    decision-making and competing at the highest level in strategic game systems.
    """
    
    result = await llm_summarize(
        text=long_text,
        max_length=50,
    )
    
    print(f"\nOriginal length: {result['original_length']} words")
    print(f"Summary length: {result['summary_length']} words")
    print(f"\nSummary:\n{result['summary']}")


async def example_entity_extraction():
    """Example 6: Extract named entities."""
    print("\n" + "="*70)
    print("Example 6: Entity Extraction")
    print("="*70)
    
    text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne 
    in Cupertino, California. The company is now led by CEO Tim Cook and 
    has offices in London, Singapore, and New York.
    """
    
    result = await llm_extract_entities(
        text=text,
        entity_types=["person", "organization", "location"],
    )
    
    print(f"\nExtracted entities:")
    import json
    print(json.dumps(result['entities'], indent=2))


async def example_translation():
    """Example 7: Translate text."""
    print("\n" + "="*70)
    print("Example 7: Translation")
    print("="*70)
    
    result = await llm_translate(
        text="Hello, how are you today? I hope you're having a great day!",
        target_language="Spanish",
    )
    
    print(f"\nTarget language: {result['target_language']}")
    print(f"Translation:\n{result['translation']}")


async def example_meta_reasoning():
    """Example 8: Meta-reasoning - using LLM to reason about using another LLM."""
    print("\n" + "="*70)
    print("Example 8: Meta-Reasoning Workflow")
    print("="*70)
    
    # First LLM decides which model and parameters to use for a task
    decision_result = await llm_generate(
        user_prompt="""
        I need to write a creative short story about space exploration.
        Which LLM model and parameters would be best for this task?
        Consider: openai:gpt-4o, mistral:mistral-large-latest
        Respond with just the model name and suggested temperature.
        """,
        system_prompt="You are an AI model selection expert.",
        temperature=0.3,  # Low temperature for consistent recommendations
    )
    
    print(f"\nModel recommendation:\n{decision_result['content']}")
    
    # Based on the recommendation, generate the story
    # (In practice, you'd parse the recommendation; here we'll just use a fixed choice)
    story_result = await llm_generate(
        user_prompt="Write a creative short story about space exploration (2-3 paragraphs).",
        model="mistral:mistral-large-latest",
        temperature=0.9,  # High temperature for creativity
    )
    
    print(f"\n\nGenerated story:\n{story_result['content']}")


async def example_with_agent():
    """Example 9: Using LLM gateway tools with an agent."""
    print("\n" + "="*70)
    print("Example 9: Agent with LLM Gateway Tools")
    print("="*70)
    
    try:
        from ai_core.interfaces.tool_registry import ToolRegistry
        from ai_core.interfaces.tool_loader import ToolLoader
        from ai_agents.components.base_agent import BaseAgent
        
        # Create a tool registry
        registry = ToolRegistry()
        
        # Register all LLM gateway tools
        tool_configs = get_all_llm_gateway_tools()
        for config in tool_configs:
            registry.register_direct_tool(
                name=config["name"],
                function=config["function"],
                description=config["description"],
                json_schema=config["json_schema"],
                category=config.get("category"),
                tags=config.get("tags", []),
            )
        
        print(f"\nRegistered {len(tool_configs)} LLM gateway tools")
        
        # Create an agent with the tools
        agent = BaseAgent(
            name="meta_reasoning_agent",
            model="openai:gpt-4o",
            system_prompt="""
            You are a meta-reasoning agent with access to various LLM tools.
            You can delegate tasks to specialized models, compare outputs, 
            and use different LLMs for different subtasks.
            """,
            tool_registry=registry,
        )
        
        # Run the agent with a meta-reasoning task
        result = await agent.arun(
            """
            I need to:
            1. Summarize a technical article about quantum computing
            2. Extract key entities (people, organizations, concepts)
            3. Translate the summary to French
            
            Please plan and execute this workflow using the available LLM tools.
            Use the sample text: 'Quantum computing is a revolutionary technology 
            being developed by IBM, Google, and Microsoft. Researchers like John Preskill 
            and Michelle Simmons are leading the field.'
            """
        )
        
        print(f"\nAgent response:\n{result.output_text}")
        
    except ImportError as e:
        print(f"\nSkipping agent example - required components not available: {e}")


async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("LLM Gateway Tools Examples")
    print("="*70)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("MISTRAL_API_KEY"):
        print("\n⚠️  Warning: No API keys found!")
        print("Please set OPENAI_API_KEY or MISTRAL_API_KEY environment variable.")
        print("Examples will fail without valid API keys.\n")
        return
    
    try:
        # Run basic examples
        await example_basic_generation()
        await example_structured_output()
        await example_token_counting()
        
        # Multi-model comparison (requires multiple API keys)
        if os.getenv("OPENAI_API_KEY") and os.getenv("MISTRAL_API_KEY"):
            await example_multi_model_comparison()
        else:
            print("\nSkipping multi-model comparison (requires both OpenAI and Mistral API keys)")
        
        # Specialized tools
        await example_summarization()
        await example_entity_extraction()
        await example_translation()
        
        # Advanced examples
        await example_meta_reasoning()
        await example_with_agent()
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())


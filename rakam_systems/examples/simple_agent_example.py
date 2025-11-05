"""
Simple example showing the most basic usage of PydanticAIAgent.
"""
import asyncio
import dotenv

dotenv.load_dotenv()

from ai_agents.components import PydanticAIAgent
from ai_core.interfaces import ModelSettings

async def main():
    # Create a simple agent without tools
    agent = PydanticAIAgent(
        name="simple_agent",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant that provides concise answers.",
    )
    
    # Example 1: Simple question
    print("=== Example 1: Simple Question ===")
    result = await agent.arun("What is 2 + 2?")
    print(f"Answer: {result.output_text}\n")
    
    # Example 2: With model settings
    print("=== Example 2: With Temperature Setting ===")
    result = await agent.arun(
        "Tell me a creative fact about space.",
        model_settings=ModelSettings(temperature=0.9, max_tokens=100),
    )
    print(f"Answer: {result.output_text}\n")
    
    # Example 3: String input (simplified API)
    print("=== Example 3: Direct String Input ===")
    result = await agent.arun("Explain Python in one sentence.")
    print(f"Answer: {result.output_text}\n")
    
    # Example 4: Streaming response
    print("=== Example 4: Streaming Response ===")
    print("Answer: ", end='', flush=True)
    async for chunk in agent.astream("Count from 1 to 5 slowly."):
        print(chunk, end='', flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())


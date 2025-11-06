"""
Example showing how to use the BaseAgent with tools.
This example demonstrates parallel tool calls and model settings.
"""
import asyncio
from datetime import datetime
from typing import Any
import dotenv

dotenv.load_dotenv()

# Import from our framework
from ai_agents.components import BaseAgent
from ai_core.interfaces import ModelSettings, Tool

# Define dummy dependency type
class Deps:
    pass

# Define tool functions
async def get_number_a() -> int:
    """Get the first number (value: 1)"""
    print(f"[{datetime.now().time()}] Starting tool A")
    await asyncio.sleep(2)
    print(f"[{datetime.now().time()}] Finished tool A")
    return 1

async def get_number_b() -> int:
    """Get the second number (value: 2)"""
    print(f"[{datetime.now().time()}] Starting tool B")
    await asyncio.sleep(2)
    print(f"[{datetime.now().time()}] Finished tool B")
    return 2

async def get_number_c() -> int:
    """Get the third number (value: 3)"""
    print(f"[{datetime.now().time()}] Starting tool C")
    await asyncio.sleep(1.5)
    print(f"[{datetime.now().time()}] Finished tool C")
    return 3

async def get_number_d() -> int:
    """Get the fourth number (value: 4)"""
    print(f"[{datetime.now().time()}] Starting tool D")
    await asyncio.sleep(1)
    print(f"[{datetime.now().time()}] Finished tool D")
    return 4

async def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers together"""
    print(f"[{datetime.now().time()}] Starting multiply({x}, {y})")
    await asyncio.sleep(0.5)
    result = x * y
    print(f"[{datetime.now().time()}] Finished multiply = {result}")
    return result

async def get_user_info() -> dict:
    """Get user information"""
    print(f"[{datetime.now().time()}] Starting get_user_info")
    await asyncio.sleep(1.5)
    print(f"[{datetime.now().time()}] Finished get_user_info")
    return {"name": "Alice", "age": 30, "city": "New York"}

# Define agent with Tool.from_schema approach
agent = BaseAgent(
    name="example_agent",
    model="openai:gpt-4o",
    deps_type=Deps,
    system_prompt="You can call tools to get numbers.",
    tools=[
        Tool.from_schema(
            function=get_number_a,
            name='get_number_a',
            description='Get the first number (value: 1)',
            json_schema={
                'additionalProperties': False,
                'properties': {},
                'type': 'object',
            },
            takes_ctx=False,
        ),
        Tool.from_schema(
            function=get_number_b,
            name='get_number_b',
            description='Get the second number (value: 2)',
            json_schema={
                'additionalProperties': False,
                'properties': {},
                'type': 'object',
            },
            takes_ctx=False,
        ),
        Tool.from_schema(
            function=get_number_c,
            name='get_number_c',
            description='Get the third number (value: 3)',
            json_schema={
                'additionalProperties': False,
                'properties': {},
                'type': 'object',
            },
            takes_ctx=False,
        ),
        Tool.from_schema(
            function=get_number_d,
            name='get_number_d',
            description='Get the fourth number (value: 4)',
            json_schema={
                'additionalProperties': False,
                'properties': {},
                'type': 'object',
            },
            takes_ctx=False,
        ),
        Tool.from_schema(
            function=multiply_numbers,
            name='multiply_numbers',
            description='Multiply two numbers together',
            json_schema={
                'additionalProperties': False,
                'properties': {
                    'x': {'description': 'the first number', 'type': 'integer'},
                    'y': {'description': 'the second number', 'type': 'integer'},
                },
                'required': ['x', 'y'],
                'type': 'object',
            },
            takes_ctx=False,
        ),
        Tool.from_schema(
            function=get_user_info,
            name='get_user_info',
            description='Get user information',
            json_schema={
                'additionalProperties': False,
                'properties': {},
                'type': 'object',
            },
            takes_ctx=False,
        ),
    ],
)

async def main():
    # --- Test 1: Simple parallel test ---
    print("\n=== Test 1: Simple parallel (A + B) ===")
    print("--- With parallel_tool_calls=True ---")
    start = datetime.now()
    result = await agent.arun(
        "Call get_number_a and get_number_b, then sum their results.",
        deps=Deps(),
        model_settings=ModelSettings(parallel_tool_calls=True),
    )
    print("Result:", result.output_text)
    print("Elapsed:", datetime.now() - start)

    print("\n--- With parallel_tool_calls=False ---")
    start = datetime.now()
    result = await agent.arun(
        "Call get_number_a and get_number_b, then sum their results.",
        deps=Deps(),
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    print("Result:", result.output_text)
    print("Elapsed:", datetime.now() - start)

    # --- Test 2: Multiple tools in parallel ---
    print("\n\n=== Test 2: Four tools (A + B + C + D) ===")
    print("--- With parallel_tool_calls=True ---")
    start = datetime.now()
    result = await agent.arun(
        "Call all four number functions (A, B, C, D) and sum all their results.",
        deps=Deps(),
        model_settings=ModelSettings(parallel_tool_calls=True),
    )
    print("Result:", result.output_text)
    print("Elapsed:", datetime.now() - start)

    print("\n--- With parallel_tool_calls=False ---")
    start = datetime.now()
    result = await agent.arun(
        "Call all four number functions (A, B, C, D) and sum all their results.",
        deps=Deps(),
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    print("Result:", result.output_text)
    print("Elapsed:", datetime.now() - start)

    # --- Test 3: Complex task with dependencies ---
    print("\n\n=== Test 3: Complex task - get numbers and multiply ===")
    print("--- With parallel_tool_calls=True ---")
    start = datetime.now()
    result = await agent.arun(
        "Get number A and number B, then multiply them together using the multiply_numbers tool.",
        deps=Deps(),
        model_settings=ModelSettings(parallel_tool_calls=True),
    )
    print("Result:", result.output_text)
    print("Elapsed:", datetime.now() - start)

    print("\n--- With parallel_tool_calls=False ---")
    start = datetime.now()
    result = await agent.arun(
        "Get number A and number B, then multiply them together using the multiply_numbers tool.",
        deps=Deps(),
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    print("Result:", result.output_text)
    print("Elapsed:", datetime.now() - start)

    # --- Test 4: Mixed independent tasks ---
    print("\n\n=== Test 4: Mixed tasks - numbers and user info ===")
    print("--- With parallel_tool_calls=True ---")
    start = datetime.now()
    result = await agent.arun(
        "Get number A, number C, and user info. Tell me all the information you gathered.",
        deps=Deps(),
        model_settings=ModelSettings(parallel_tool_calls=True),
    )
    print("Result:", result.output_text)
    print("Elapsed:", datetime.now() - start)

    print("\n--- With parallel_tool_calls=False ---")
    start = datetime.now()
    result = await agent.arun(
        "Get number A, number C, and user info. Tell me all the information you gathered.",
        deps=Deps(),
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    print("Result:", result.output_text)
    print("Elapsed:", datetime.now() - start)

if __name__ == "__main__":
    asyncio.run(main())


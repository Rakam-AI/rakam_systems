---
title: Agents
---

# Building AI Agents

This section covers the main usage patterns of `BaseAgent`: basic inference, streaming, model settings, tools, structured output, RAG pipelines, and YAML configuration.

## Run a basic agent

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()

from rakam_systems_agent import BaseAgent

async def main():
    agent = BaseAgent(
        name="my_assistant",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant."
    )
    result = await agent.arun("What is Python?")
    print(result.output_text)

asyncio.run(main())
```

## Stream responses

```python
async def main():
    agent = BaseAgent(name="stream_agent", model="openai:gpt-4o")

    print("Response: ", end="", flush=True)
    async for chunk in agent.astream("Tell me a short story."):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

## Customize model settings

```python
from rakam_systems_core.interfaces import ModelSettings

agent = BaseAgent(
    name="creative",
    model="openai:gpt-4o",
    system_prompt="You are a creative writer."
)
result = await agent.arun(
    "Write a haiku about programming.",
    model_settings=ModelSettings(temperature=0.9, max_tokens=100)
)
```

## Add tools

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_core.interfaces.tool import ToolComponent

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 22°C, Sunny"

weather_tool = ToolComponent.from_function(
    function=get_weather,
    name="get_weather",
    description="Get current weather for a city",
    json_schema={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"]
    }
)

async def main():
    agent = BaseAgent(
        name="tool_agent",
        model="openai:gpt-4o",
        tools=[weather_tool]
    )
    result = await agent.arun("What's the weather in Paris?")
    print(result.output_text)

asyncio.run(main())
```

## Use structured output

```python
import asyncio
from pydantic import BaseModel, Field
from rakam_systems_agent import BaseAgent

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(ge=0, le=10)
    summary: str = Field(description="Brief summary")
    recommended: bool

async def main():
    agent = BaseAgent(
        name="critic",
        model="openai:gpt-4o",
        output_type=MovieReview
    )
    result = await agent.arun("Review the movie 'Inception'")

    review: MovieReview = result.output
    print(f"Rating: {review.rating}/10")
    print(f"Recommended: {'Yes' if review.recommended else 'No'}")

asyncio.run(main())
```

## Build a RAG pipeline

Combine agents and vector store for question-answering over your documents:

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_vectorstore import FaissStore, Node, NodeMetadata
from rakam_systems_core.interfaces.tool import ToolComponent

# 1. Create vector store with your documents
store = FaissStore(name="kb", base_index_path="./kb_index",
                   embedding_model="Snowflake/snowflake-arctic-embed-m", initialising=True)

kb_nodes = [
    Node(content="Our company was founded in 2020.", metadata=NodeMetadata(source_file_uuid="info", position=0)),
    Node(content="We offer AI Assistant at $99/month.", metadata=NodeMetadata(source_file_uuid="info", position=1)),
]
store.create_collection_from_nodes("knowledge", kb_nodes)

# 2. Create search tool
def search_kb(query: str) -> str:
    results, _ = store.search(collection_name="knowledge", query=query, number=3)
    return "\n".join([content for _, (_, content, _) in results.items()])

search_tool = ToolComponent.from_function(
    function=search_kb, name="search_kb",
    description="Search company knowledge base",
    json_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    }
)

# 3. Create RAG agent
async def main():
    agent = BaseAgent(
        name="rag_agent",
        model="openai:gpt-4o",
        system_prompt="Use search_kb tool to find information. Answer based on retrieved docs.",
        tools=[search_tool]
    )
    result = await agent.arun("How much does your product cost?")
    print(result.output_text)  # "$99/month"

asyncio.run(main())
```


## Configure an agent with YAML

Create agents from config files — no code changes needed to switch models or prompts:

**`config/agent.yaml`:**

```yaml
version: "1.0"

agents:
  assistant:
    name: "assistant"
    llm_config:
      model: "openai:gpt-4o"
      temperature: 0.7
    system_prompt: "You are a helpful assistant."
```

**Use in code:**

```python
import asyncio
from rakam_systems_core.config_loader import ConfigurationLoader

loader = ConfigurationLoader()
config = loader.load_from_yaml("config/agent.yaml")
agent = loader.create_agent("assistant", config)

asyncio.run(agent.arun("Hello!"))
```

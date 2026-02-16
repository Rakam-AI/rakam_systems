# Quick Start Guide

Get up and running with Rakam Systems in 5 minutes!

## TL;DR - The Simplest Example

```python
import asyncio
from rakam_systems_agent import BaseAgent

async def main():
    agent = BaseAgent(name="assistant", model="openai:gpt-4o")
    result = await agent.arun("What is Python?")
    print(result.output_text)

asyncio.run(main())
```

> **Prerequisites:** Python 3.10+, OpenAI API key

---

## Table of Contents

- [Installation](#installation)
- [Your First Agent](#your-first-agent)
- [Common Use Cases](#common-use-cases)
  - [With Tools](#with-tools)
  - [Structured Output](#structured-output)
  - [Chat History](#chat-history)
  - [Vector Store](#vector-store)
  - [RAG Pipeline](#rag-pipeline)
- [Configuration (YAML)](#configuration-yaml)
- [Next Steps](#next-steps)

---

## Installation

```bash
# Core package
pip install rakam-system-core

# For AI agents
pip install rakam-systems-agent[all]

# For vector search
pip install rakam-systems-vectorstore[all]

# Or everything at once
pip install rakam-systems
```

Set your API key:

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="sk-your-api-key"

# Option 2: .env file (recommended)
# Create .env with: OPENAI_API_KEY=sk-your-api-key
# Then add to your code: from dotenv import load_dotenv; load_dotenv()
```

---

## Your First Agent

### Basic Agent

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

### With Streaming

```python
async def main():
    agent = BaseAgent(name="stream_agent", model="openai:gpt-4o")

    print("Response: ", end="", flush=True)
    async for chunk in agent.astream("Tell me a short story."):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

### With Custom Settings

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

---

## Common Use Cases

### With Tools

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_core.interfaces.tool import ToolComponent

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 22°C, Sunny"

# Create tool from function
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

### Structured Output

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

### Chat History

Choose the right backend for your needs:

| Backend        | Use Case    | Quick Setup            |
| -------------- | ----------- | ---------------------- |
| **JSON**       | Prototyping | File-based, no setup   |
| **SQLite**     | Local dev   | Single file, no server |
| **PostgreSQL** | Production  | Scalable, concurrent   |

**JSON (simplest):**

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_agent.components.chat_history import JSONChatHistory

async def main():
    history = JSONChatHistory(config={"storage_path": "./chat.json"})
    agent = BaseAgent(name="chatty", model="openai:gpt-4o")

    chat_id = "user_123"

    # First message
    messages = history.get_message_history(chat_id)
    result = await agent.arun("My name is Alice!", message_history=messages)
    history.save_messages(chat_id, result.all_messages())

    # Second message (remembers context)
    messages = history.get_message_history(chat_id)
    result = await agent.arun("What's my name?", message_history=messages)
    print(result.output_text)  # "Your name is Alice!"

asyncio.run(main())
```

**PostgreSQL (production):**

```python
# Uses environment variables: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, etc.
history = PostgresChatHistory()  # No config needed if env vars set
```

### Vector Store

**FAISS (in-memory, for development):**

```python
from rakam_systems_vectorstore import FaissStore, Node, NodeMetadata

store = FaissStore(
    name="my_store",
    base_index_path="./indexes",
    embedding_model="Snowflake/snowflake-arctic-embed-m",
    initialising=True
)

# Add documents
nodes = [
    Node(content="Python is a programming language.",
         metadata=NodeMetadata(source_file_uuid="doc1", position=0)),
    Node(content="Machine learning is AI subset.",
         metadata=NodeMetadata(source_file_uuid="doc1", position=1)),
]
store.create_collection_from_nodes("docs", nodes)

# Search
results, _ = store.search(collection_name="docs", query="What is Python?", number=3)
for _, (_, content, dist) in results.items():
    print(f"[{dist:.4f}] {content}")
```

**PostgreSQL (production):**

```python
# Requires: pip install rakam-systems-vectorstore[all]
# Requires: PostgreSQL with pgvector extension

from rakam_systems_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

config = VectorStoreConfig(
    name="prod_store",
    embedding={"model_type": "sentence_transformer", "model_name": "Snowflake/snowflake-arctic-embed-m"}
)
# Database config via environment variables (POSTGRES_HOST, etc.)

store = ConfigurablePgVectorStore(config=config)
store.setup()
# ... use store ...
store.shutdown()
```

### RAG Pipeline

Combine agents + vector store for question-answering over your documents:

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

---

## Configuration (YAML)

Create agents from config files - no code changes needed to switch models/prompts:

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

asyncio.run(lambda: agent.arun("Hello!"))
```

---

## Next Steps

1. **Read the Full Documentation**
   - [Components Guide](components.md) - Detailed component docs
   - [Installation Guide](installation.md) - Advanced setup options

2. **Explore Examples**
   - `rakam_systems/examples/ai_agents_examples/` - Agent patterns
   - `rakam_systems/examples/ai_vectorstore_examples/` - Vector store examples

3. **Build Your App**
   - Start simple → Add tools → Scale with PostgreSQL

---

**Need Help?** [GitHub Issues](https://github.com/Rakam-AI/rakam_systems) | [Report Bugs](https://github.com/Rakam-AI/rakam_systems/issues)

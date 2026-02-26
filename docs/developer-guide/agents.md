---
title: Agent Package
---

# Agent Package

The agent package provides AI agent implementations powered by Pydantic AI. Install with `pip install rakam-systems-agent[all]` (requires core).

## BaseAgent

The main agent implementation using Pydantic AI:

```python
from rakam_systems_agent import BaseAgent
from rakam_systems_core.interfaces.agent import AgentInput, AgentOutput, ModelSettings

agent = BaseAgent(
    name="my_agent",
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
    tools=[my_tool],  # Optional tools
    output_type=MyOutputModel,  # Optional structured output
    enable_tracking=True  # Optional tracking
)

# Async inference (required for Pydantic AI)
result = await agent.arun("What is AI?")
print(result.output_text)

# With dependencies
result = await agent.arun("Hello", deps={"user_id": "123"})

# With model settings
settings = ModelSettings(temperature=0.5, max_tokens=1000)
result = await agent.arun("Explain quantum computing", model_settings=settings)

# Streaming
async for chunk in agent.astream("Tell me a story"):
    print(chunk, end="")
```

### Dynamic System Prompts

Dynamic system prompts allow you to inject context at runtime based on current state, user information, or external data:

```python
from datetime import date, datetime
from pydantic_ai import RunContext

agent = BaseAgent(
    name="dynamic_agent",
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Method 1: Decorator syntax
@agent.dynamic_system_prompt
def add_date() -> str:
    """Add current date to system prompt."""
    return f"Today's date is {date.today().strftime('%B %d, %Y')}."

@agent.dynamic_system_prompt
def add_user_context(ctx: RunContext[dict]) -> str:
    """Add user-specific context from dependencies."""
    if ctx.deps and "user_name" in ctx.deps:
        return f"You are assisting {ctx.deps['user_name']}."
    return ""

# Method 2: Direct registration
def add_time_context() -> str:
    """Add current time to system prompt."""
    return f"Current time: {datetime.now().strftime('%H:%M:%S')}"

agent.add_dynamic_system_prompt(add_time_context)

# Method 3: Async dynamic prompts
@agent.dynamic_system_prompt
async def fetch_external_context(ctx: RunContext[dict]) -> str:
    """Fetch and add external context asynchronously."""
    # Example: fetch from API or database
    import asyncio
    await asyncio.sleep(0.1)
    return "Additional context from external source."

# Usage with dependencies
result = await agent.arun(
    "What day is it?",
    deps={"user_name": "Alice", "user_id": "123"}
)
```

## LLM Gateways

### OpenAI Gateway

```python
from rakam_systems_agent import OpenAIGateway, LLMRequest

gateway = OpenAIGateway(
    model="gpt-4o",
    api_key="...",  # Or use OPENAI_API_KEY env var
    default_temperature=0.7
)

# Text generation
request = LLMRequest(
    system_prompt="You are a helpful assistant",
    user_prompt="What is AI?",
    temperature=0.7
)
response = gateway.generate(request)
print(response.content)

# Structured output
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

result = gateway.generate_structured(request, Answer)
print(result.answer, result.confidence)

# Streaming
for chunk in gateway.stream(request):
    print(chunk, end="")

# Token counting
token_count = gateway.count_tokens("Hello, world!")
```

### Mistral Gateway

```python
from rakam_systems_agent import MistralGateway

gateway = MistralGateway(
    model="mistral-large-latest",
    api_key="..."  # Or use MISTRAL_API_KEY env var
)
```

### Gateway Factory

```python
from rakam_systems_agent import LLMGatewayFactory, get_llm_gateway

# Using factory
gateway = LLMGatewayFactory.create(
    provider="openai",
    model="gpt-4o",
    api_key="..."
)

# Using convenience function
gateway = get_llm_gateway(provider="openai", model="gpt-4o")
```

## Chat History

### JSON Chat History

```python
from rakam_systems_agent.components.chat_history import JSONChatHistory

history = JSONChatHistory(config={"storage_path": "./chat_history.json"})

# Add messages
history.add_message("chat123", {"role": "user", "content": "Hello!"})
history.add_message("chat123", {"role": "assistant", "content": "Hi there!"})

# Get history
messages = history.get_chat_history("chat123")
readable = history.get_readable_chat_history("chat123")

# Pydantic AI integration
message_history = history.get_message_history("chat123")
result = await agent.run("Hello", message_history=message_history)
history.save_messages("chat123", result.all_messages())

# Manage chats
all_chats = history.get_all_chat_ids()
history.delete_chat_history("chat123")
history.clear_all()
```

### SQL Chat History (SQLite)

```python
from rakam_systems_agent.components.chat_history import SQLChatHistory

history = SQLChatHistory(config={"db_path": "./chat_history.db"})

# Same API as JSON Chat History
history.add_message("chat123", {"role": "user", "content": "Hello!"})
history.add_message("chat123", {"role": "assistant", "content": "Hi there!"})

# Get history
messages = history.get_chat_history("chat123")

# Pydantic AI integration
message_history = history.get_message_history("chat123")
result = await agent.run("Hello", message_history=message_history)
history.save_messages("chat123", result.all_messages())
```

### PostgreSQL Chat History

For production deployments with PostgreSQL-backed storage:

```python
from rakam_systems_agent.components.chat_history import PostgresChatHistory

# Configuration
history = PostgresChatHistory(config={
    "host": "localhost",
    "port": 5432,
    "database": "chat_db",
    "user": "postgres",
    "password": "postgres"
})

# Or use environment variables (POSTGRES_HOST, POSTGRES_PORT, etc.)
history = PostgresChatHistory()

# Same API as other chat history backends
history.add_message("chat123", {"role": "user", "content": "Hello!"})
history.add_message("chat123", {"role": "assistant", "content": "Hi there!"})

# Get history
messages = history.get_chat_history("chat123")
readable = history.get_readable_chat_history("chat123")

# Pydantic AI integration
message_history = history.get_message_history("chat123")
result = await agent.run("Hello", message_history=message_history)
history.save_messages("chat123", result.all_messages())

# Manage chats
all_chats = history.get_all_chat_ids()
history.delete_chat_history("chat123")
history.clear_all()

# Cleanup
history.shutdown()
```

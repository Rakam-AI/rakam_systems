---
title: Manage chat history
---

# Manage chat history

All chat history backends share the same API. Choose a backend based on your deployment needs.

## JSON (development)

File-based storage, suitable for local development and prototyping:

```python
from rakam_systems_agent.components.chat_history import JSONChatHistory

history = JSONChatHistory(config={"storage_path": "./chat_history.json"})

# Add messages
history.add_message("chat123", {"role": "user", "content": "Hello!"})
history.add_message("chat123", {"role": "assistant", "content": "Hi there!"})

# Get history
messages = history.get_chat_history("chat123")
readable = history.get_readable_chat_history("chat123")

# Manage chats
all_chats = history.get_all_chat_ids()
history.delete_chat_history("chat123")
history.clear_all()
```

## SQLite (lightweight production)

Database-backed storage without external dependencies:

```python
from rakam_systems_agent.components.chat_history import SQLChatHistory

history = SQLChatHistory(config={"db_path": "./chat_history.db"})

# Same API as JSON Chat History
history.add_message("chat123", {"role": "user", "content": "Hello!"})
messages = history.get_chat_history("chat123")
```

## PostgreSQL (production)

For production deployments with PostgreSQL-backed storage:

```python
from rakam_systems_agent.components.chat_history import PostgresChatHistory

# Explicit configuration
history = PostgresChatHistory(config={
    "host": "localhost",
    "port": 5432,
    "database": "chat_db",
    "user": "postgres",
    "password": "postgres"
})

# Or use environment variables (POSTGRES_HOST, POSTGRES_PORT, etc.)
history = PostgresChatHistory()

# Same API as other backends
history.add_message("chat123", {"role": "user", "content": "Hello!"})
messages = history.get_chat_history("chat123")

# Cleanup
history.shutdown()
```

## Pydantic AI integration

All backends support Pydantic AI's message history format:

```python
# Get message history in Pydantic AI format
message_history = history.get_message_history("chat123")
result = await agent.run("Hello", message_history=message_history)

# Save messages back
history.save_messages("chat123", result.all_messages())
```

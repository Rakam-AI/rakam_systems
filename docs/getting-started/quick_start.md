# Quick Start Guide

Get up and running with Rakam Systems in 5 minutes!

## 📑 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start: AI Agent](#quick-start-ai-agent)
- [Quick Start: Agent with Tools](#quick-start-agent-with-tools)
- [Quick Start: Structured Output](#quick-start-structured-output)
- [Quick Start: Configurable Agent](#quick-start-configurable-agent)
- [Quick Start: Vector Store](#quick-start-vector-store)
- [Quick Start: Configurable Vector Store](#quick-start-configurable-vector-store)
- [Quick Start: RAG Pipeline](#quick-start-rag-pipeline)
- [Quick Start: LLM Gateway](#quick-start-llm-gateway)
- [Quick Start: Document Loading](#quick-start-document-loading)
- [Quick Start: Configuration from YAML](#quick-start-configuration-from-yaml)
- [Common Patterns](#common-patterns)
- [Next Steps](#next-steps)

---

## Prerequisites

- Python 3.10 or higher
- OpenAI API key (for most examples)
- pip package manager

---

## Installation

```bash
# Install core package (required)
pip install -e ./rakam-system-core

# Install agent package for AI agents
pip install -e ./rakam-system-agent

# Install vectorstore package for vector search
pip install -e ./rakam-system-vectorstore

# Or install everything at once
pip install -e ./rakam-system-core ./rakam-system-agent ./rakam-system-vectorstore
```

Set your API key:

```bash
export OPENAI_API_KEY="sk-your-api-key"
```

---

## Quick Start: AI Agent

Create a simple AI agent in just a few lines:

```python
import asyncio
from dotenv import load_dotenv

load_dotenv()

from rakam_systems_agent import BaseAgent

async def main():
    # Create an agent
    agent = BaseAgent(
        name="my_assistant",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant that provides concise answers."
    )

    # Ask a question
    result = await agent.arun("What is Python?")
    print(result.output_text)

asyncio.run(main())
```

### With Streaming

```python
import asyncio
from rakam_systems_agent import BaseAgent

async def main():
    agent = BaseAgent(
        name="streaming_agent",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant."
    )

    # Stream the response
    print("Response: ", end="", flush=True)
    async for chunk in agent.astream("Tell me a short story about a robot."):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

### With Model Settings

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_core.interfaces import ModelSettings

async def main():
    agent = BaseAgent(
        name="creative_agent",
        model="openai:gpt-4o",
        system_prompt="You are a creative writer."
    )

    # Use custom temperature and max tokens
    result = await agent.arun(
        "Write a haiku about programming.",
        model_settings=ModelSettings(temperature=0.9, max_tokens=100)
    )
    print(result.output_text)

asyncio.run(main())
```

---

## Quick Start: Agent with Tools

Create an agent that can use tools:

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_core.interfaces.tool import ToolComponent

# Define a tool function
def get_weather(city: str, units: str = "celsius") -> str:
    """Get weather for a city (mock implementation)."""
    return f"Weather in {city}: 22°{'C' if units == 'celsius' else 'F'}, Sunny"

def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Create tool components
weather_tool = ToolComponent.from_function(
    function=get_weather,
    name="get_weather",
    description="Get the current weather for a city",
    json_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["city"],
        "additionalProperties": False
    }
)

calculator_tool = ToolComponent.from_function(
    function=calculate,
    name="calculator",
    description="Calculate a mathematical expression",
    json_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}
        },
        "required": ["expression"],
        "additionalProperties": False
    }
)

async def main():
    # Create agent with tools
    agent = BaseAgent(
        name="tool_agent",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant with access to tools.",
        tools=[weather_tool, calculator_tool]
    )

    # Ask questions that use tools
    result = await agent.arun("What's the weather in Paris?")
    print(f"Weather: {result.output_text}\n")

    result = await agent.arun("What is 25 * 4 + 100?")
    print(f"Calculation: {result.output_text}")

asyncio.run(main())
```

---

## Quick Start: Structured Output

Get structured, typed responses from your agent:

```python
import asyncio
from pydantic import BaseModel, Field
from rakam_systems_agent import BaseAgent

# Define output structure
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(ge=0, le=10, description="Rating out of 10")
    summary: str = Field(description="Brief summary")
    pros: list[str] = Field(description="List of positive aspects")
    cons: list[str] = Field(description="List of negative aspects")
    recommended: bool = Field(description="Whether to recommend")

async def main():
    # Create agent with structured output
    agent = BaseAgent(
        name="movie_critic",
        model="openai:gpt-4o",
        system_prompt="You are a movie critic. Analyze movies thoroughly.",
        output_type=MovieReview  # Enforces structured output
    )

    result = await agent.arun("Review the movie 'Inception' by Christopher Nolan")

    # Access typed output
    review: MovieReview = result.output
    print(f"Title: {review.title}")
    print(f"Rating: {review.rating}/10")
    print(f"Summary: {review.summary}")
    print(f"Pros: {', '.join(review.pros)}")
    print(f"Cons: {', '.join(review.cons)}")
    print(f"Recommended: {'Yes' if review.recommended else 'No'}")

asyncio.run(main())
```

---

## Quick Start: Chat History

Maintain conversation context across multiple interactions:

### JSON Chat History (File-based)

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_agent.components.chat_history import JSONChatHistory

async def main():
    # Initialize chat history
    history = JSONChatHistory(config={
        "storage_path": "./chat_history.json"
    })

    # Create agent
    agent = BaseAgent(
        name="chat_agent",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant with memory."
    )

    chat_id = "user_123"

    # First conversation
    message_history = history.get_message_history(chat_id)
    result = await agent.arun(
        "My name is Alice and I love Python programming.",
        message_history=message_history
    )
    history.save_messages(chat_id, result.all_messages())
    print(f"Agent: {result.output_text}\n")

    # Second conversation (agent remembers)
    message_history = history.get_message_history(chat_id)
    result = await agent.arun(
        "What's my name and what do I love?",
        message_history=message_history
    )
    history.save_messages(chat_id, result.all_messages())
    print(f"Agent: {result.output_text}")

asyncio.run(main())
```

### PostgreSQL Chat History (Production)

For production deployments with PostgreSQL:

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_agent.components.chat_history import PostgresChatHistory

async def main():
    # Initialize PostgreSQL chat history
    history = PostgresChatHistory(config={
        "host": "localhost",
        "port": 5432,
        "database": "chat_db",
        "user": "postgres",
        "password": "postgres"
    })

    # Or use environment variables (POSTGRES_HOST, POSTGRES_PORT, etc.)
    # history = PostgresChatHistory()

    # Create agent
    agent = BaseAgent(
        name="chat_agent",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant."
    )

    chat_id = "user_456"

    # Get existing history
    message_history = history.get_message_history(chat_id)

    # Continue conversation
    result = await agent.arun(
        "Tell me about machine learning.",
        message_history=message_history
    )

    # Save new messages
    history.save_messages(chat_id, result.all_messages())
    print(f"Agent: {result.output_text}")

    # Get readable history
    readable = history.get_readable_chat_history(chat_id)
    print(f"\nConversation History:\n{readable}")

    # Cleanup
    history.shutdown()

asyncio.run(main())
```

### SQLite Chat History (Local Database)

For local development with SQLite:

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_agent.components.chat_history import SQLChatHistory

async def main():
    # Initialize SQLite chat history
    history = SQLChatHistory(config={
        "db_path": "./chat_history.db"
    })

    agent = BaseAgent(
        name="chat_agent",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant."
    )

    chat_id = "user_789"

    # Use with agent
    message_history = history.get_message_history(chat_id)
    result = await agent.arun(
        "Hello! How are you?",
        message_history=message_history
    )
    history.save_messages(chat_id, result.all_messages())

    print(f"Agent: {result.output_text}")

asyncio.run(main())
```

**Choosing a Backend:**

- **JSONChatHistory**: Simple file-based storage, good for prototyping
- **SQLChatHistory**: SQLite database, good for local development
- **PostgresChatHistory**: Production-ready, scalable, concurrent access

---

## Quick Start: Configurable Agent

**The Power of Configuration-First Design**: Create agents from YAML configuration files and change behavior without touching your code! This is perfect for:

- **Production deployments**: Tune parameters without redeploying
- **A/B testing**: Test different models/settings by swapping config files
- **Environment management**: Different configs for dev/staging/production
- **Rapid iteration**: Experiment with prompts, tools, and settings instantly

### Agent Configuration File

Create `config/agent_config.yaml`:

```yaml
version: "1.0"

# Define reusable prompts
prompts:
  customer_support:
    name: "customer_support"
    description: "Friendly customer support specialist"
    system_prompt: |
      You are a friendly and empathetic customer support specialist.
      Your goal is to help customers solve their problems efficiently.

      Guidelines:
      - Listen actively to customer concerns
      - Provide clear, step-by-step solutions
      - Always maintain a professional yet warm tone
    skills:
      - "Problem solving"
      - "Customer communication"
    tags:
      - "support"

  code_assistant:
    name: "code_assistant"
    description: "Expert programming assistant"
    system_prompt: |
      You are an expert programming assistant with deep knowledge of:
      - Multiple programming languages (Python, JavaScript, TypeScript, etc.)
      - Software design patterns and best practices
      - Code optimization and debugging

      When generating code:
      1. Follow best practices and conventions
      2. Include comments and documentation
      3. Consider edge cases and error handling

# Define reusable tools
tools:
  get_weather:
    name: "get_weather"
    type: "direct"
    module: "rakam_system_agent.components.tools.example_tools"
    function: "get_current_weather"
    description: "Get the current weather for a location"
    category: "utility"
    tags: ["weather", "external"]
    json_schema:
      type: "object"
      properties:
        location:
          type: "string"
          description: "City name or location"
        units:
          type: "string"
          enum: ["celsius", "fahrenheit"]
          default: "celsius"
      required: ["location"]
      additionalProperties: false

  analyze_sentiment:
    name: "analyze_sentiment"
    type: "direct"
    module: "rakam_system_agent.components.tools.example_tools"
    function: "analyze_sentiment"
    description: "Analyze the sentiment of text"
    category: "nlp"
    json_schema:
      type: "object"
      properties:
        text:
          type: "string"
          description: "Text to analyze"
      required: ["text"]
      additionalProperties: false

# Define agents
agents:
  support_agent:
    name: "support_agent"
    description: "Customer support specialist"

    # Model configuration
    llm_config:
      model: "openai:gpt-4o"
      temperature: 0.7
      max_tokens: 2000
      parallel_tool_calls: true

    # Reference to prompt library
    prompt_config: "customer_support"

    # Reference to tools library
    tools:
      - "get_weather"
      - "analyze_sentiment"

    # Enable input/output tracking
    enable_tracking: true
    tracking_output_dir: "./agent_tracking/support"

    # Additional metadata
    metadata:
      version: "1.0"
      department: "customer_support"

  # Agent with structured output defined inline
  sql_agent:
    name: "sql_agent"
    description: "SQL query assistant with structured output"

    llm_config:
      model: "openai:gpt-4o"
      temperature: 0.2 # Low temperature for consistent output
      max_tokens: 3000

    prompt_config: "code_assistant"
    tools: []

    # Define structured output directly in YAML (no Python class needed!)
    output_type:
      name: "SQLAgentOutput"
      description: "Structured output for SQL queries"
      fields:
        answer:
          type: str
          description: "The answer to the user's question"
        sql_query:
          type: str
          description: "The generated SQL query"
          default: ""
        explanation:
          type: str
          description: "Explanation of the query"
        tables_used:
          type: list
          description: "List of tables referenced"
          default_factory: list

    enable_tracking: true
```

### Loading Agents from Configuration

**Key Benefit**: Your application code stays the same - just swap config files!

```python
import asyncio
from rakam_systems_core.config_loader import ConfigurationLoader

async def main():
    # Initialize configuration loader
    loader = ConfigurationLoader()

    # Load configuration from YAML
    # Change behavior by using different config files - no code changes!
    config_file = "config/agent_config.yaml"  # or "config/agent_config_prod.yaml"
    config = loader.load_from_yaml(config_file)

    # Validate configuration (optional but recommended)
    is_valid, errors = loader.validate_config()
    if not is_valid:
        for error in errors:
            print(f"Config error: {error}")
        return

    # Create a single agent
    support_agent = loader.create_agent("support_agent", config)

    # Use the agent - behavior determined by config, not code!
    result = await support_agent.arun("What's the weather in New York?")
    print(f"Support Agent: {result.output_text}")

    # Create agent with structured output
    sql_agent = loader.create_agent("sql_agent", config)
    result = await sql_agent.arun("Write a query to find all users who signed up last month")

    # Access structured output
    print(f"Answer: {result.output.answer}")
    print(f"SQL: {result.output.sql_query}")
    print(f"Tables: {result.output.tables_used}")

    # Create all agents at once
    all_agents = loader.create_all_agents(config)
    print(f"Created {len(all_agents)} agents: {list(all_agents.keys())}")

asyncio.run(main())
```

**Example: Switching Models Without Code Changes**

```bash
# Development: Use faster, cheaper model
# config/agent_config_dev.yaml
agents:
  my_agent:
    llm_config:
      model: openai:gpt-4o-mini
      temperature: 0.7

# Production: Use more capable model
# config/agent_config_prod.yaml
agents:
  my_agent:
    llm_config:
      model: openai:gpt-4o
      temperature: 0.5

# Application code stays the same - just change config file path!
# config = loader.load_from_yaml("config/agent_config_prod.yaml")
```

### Using Tool Registry

```python
from rakam_systems_core.config_loader import ConfigurationLoader
from rakam_systems_core.interfaces.tool_registry import ToolRegistry

loader = ConfigurationLoader()
config = loader.load_from_yaml("config/agent_config.yaml")

# Get tool registry with all configured tools
registry = loader.get_tool_registry(config)

# Query tools by category
utility_tools = registry.get_tools_by_category("utility")
print(f"Utility tools: {[t.name for t in utility_tools]}")

# Query tools by tag
external_tools = registry.get_tools_by_tag("external")
print(f"External tools: {[t.name for t in external_tools]}")

# Get specific tool
weather_tool = registry.get_tool("get_weather")
print(f"Weather tool: {weather_tool.description}")
```

### Configuration Options Reference

#### Model Configuration (`llm_config`)

| Option                | Type   | Default  | Description                              |
| --------------------- | ------ | -------- | ---------------------------------------- |
| `model`               | string | required | Model identifier (e.g., `openai:gpt-4o`) |
| `temperature`         | float  | 0.7      | Creativity (0.0-1.0)                     |
| `max_tokens`          | int    | 2000     | Maximum response tokens                  |
| `parallel_tool_calls` | bool   | true     | Execute tools in parallel                |
| `extra_settings`      | dict   | {}       | Additional provider settings             |

#### Tool Configuration

| Option        | Type   | Required | Description                     |
| ------------- | ------ | -------- | ------------------------------- |
| `name`        | string | ✓        | Unique tool identifier          |
| `type`        | string | ✓        | `direct` or `mcp`               |
| `module`      | string | ✓\*      | Python module path (for direct) |
| `function`    | string | ✓\*      | Function name (for direct)      |
| `description` | string | ✓        | Tool description for LLM        |
| `json_schema` | object | ✓        | Parameter schema                |
| `category`    | string | -        | Tool category for filtering     |
| `tags`        | list   | -        | Tags for filtering              |

#### Inline Output Type

| Option        | Type   | Required | Description       |
| ------------- | ------ | -------- | ----------------- |
| `name`        | string | ✓        | Model class name  |
| `description` | string | -        | Model description |
| `fields`      | dict   | ✓        | Field definitions |

Field options: `type`, `description`, `default`, `default_factory`, `required`

---

## Quick Start: Vector Store

Create a vector store for semantic search:

### Using FAISS (In-Memory)

```python
from rakam_system_vectorstore.components.vectorstore.faiss_vector_store import FaissStore
from rakam_system_vectorstore.core import Node, NodeMetadata

# Create sample documents
documents = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand text.",
    "Vector databases store data based on semantic similarity.",
]

# Create nodes
nodes = []
for i, doc in enumerate(documents):
    metadata = NodeMetadata(
        source_file_uuid="docs_001",
        position=i,
        custom={"category": "tech"}
    )
    nodes.append(Node(content=doc, metadata=metadata))

# Initialize FAISS store
store = FaissStore(
    name="my_store",
    base_index_path="./my_indexes",
    embedding_model="Snowflake/snowflake-arctic-embed-m",
    initialising=True
)

# Create collection and add nodes
store.create_collection_from_nodes("tech_docs", nodes)

# Search
query = "What is machine learning?"
results, result_nodes = store.search(
    collection_name="tech_docs",
    query=query,
    distance_type="cosine",
    number=3
)

print(f"Query: {query}\n")
for node_id, (metadata, content, distance) in results.items():
    print(f"  [{distance:.4f}] {content}")
```

### Using PostgreSQL with pgvector

```python
import os
import django
from django.conf import settings

# Configure Django (required for PostgreSQL backend)
if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'rakam_system_vectorstore.components.vectorstore',
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.postgresql',
                'NAME': os.getenv('POSTGRES_DB', 'vectorstore_db'),
                'USER': os.getenv('POSTGRES_USER', 'postgres'),
                'PASSWORD': os.getenv('POSTGRES_PASSWORD', 'postgres'),
                'HOST': os.getenv('POSTGRES_HOST', 'localhost'),
                'PORT': os.getenv('POSTGRES_PORT', '5432'),
            }
        },
        DEFAULT_AUTO_FIELD='django.db.models.BigAutoField',
    )
    django.setup()

from rakam_system_vectorstore import (
    ConfigurablePgVectorStore,
    VectorStoreConfig,
    Node,
    NodeMetadata
)

# Create configuration
config = VectorStoreConfig(
    name="my_pg_store",
    embedding={
        "model_type": "sentence_transformer",
        "model_name": "Snowflake/snowflake-arctic-embed-m"
    },
    search={
        "similarity_metric": "cosine",
        "default_top_k": 5
    }
)

# Initialize store
store = ConfigurablePgVectorStore(config=config)
store.setup()

# Add documents
nodes = [
    Node(
        content="Python is great for data science.",
        metadata=NodeMetadata(source_file_uuid="doc1", position=0)
    ),
    Node(
        content="JavaScript runs in web browsers.",
        metadata=NodeMetadata(source_file_uuid="doc1", position=1)
    ),
]
store.add_nodes(nodes)

# Search
results = store.search("What language is good for data?", top_k=3)
for r in results:
    print(f"[{r['score']:.4f}] {r['content']}")

# Cleanup
store.shutdown()
```

---

## Quick Start: Configurable Vector Store

**Configuration-Driven Vector Storage**: The `ConfigurablePgVectorStore` lets you change your entire vector store setup without modifying code:

- **Switch embedding models**: From local to OpenAI embeddings by editing config
- **Change search algorithms**: Toggle between BM25 and ts_rank for keyword search
- **Adjust search behavior**: Change similarity metrics, hybrid search weights, etc.
- **Tune performance**: Modify batch sizes, chunk sizes, all via YAML

**Perfect for**: Testing different embedding models, optimizing search relevance, environment-specific settings

### VectorStoreConfig Overview

```python
from rakam_system_vectorstore.config import (
    VectorStoreConfig,
    EmbeddingConfig,
    DatabaseConfig,
    SearchConfig,
    IndexConfig,
    load_config
)

# Create configuration programmatically
config = VectorStoreConfig(
    name="production_store",

    # Embedding configuration
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",  # or "openai", "cohere"
        model_name="Snowflake/snowflake-arctic-embed-m",
        batch_size=128,
        normalize=True,
        # api_key loaded from env: OPENAI_API_KEY or COHERE_API_KEY
    ),

    # Database configuration (uses env vars by default)
    database=DatabaseConfig(
        host="localhost",  # or POSTGRES_HOST env var
        port=5432,         # or POSTGRES_PORT env var
        database="vectorstore_db",
        user="postgres",
        password="postgres",
    ),

    # Search configuration
    search=SearchConfig(
        similarity_metric="cosine",  # "cosine", "l2", "dot_product"
        default_top_k=5,
        enable_hybrid_search=True,   # Combine vector + keyword search
        hybrid_alpha=0.7,            # Vector weight (0.3 for keyword)
        rerank=True,
    ),

    # Indexing configuration
    index=IndexConfig(
        chunk_size=512,
        chunk_overlap=50,
        batch_insert_size=10000,
    ),

    # Additional options
    enable_caching=True,
    cache_size=1000,
    enable_logging=True,
    log_level="INFO",
)

# Validate configuration
config.validate()
```

### Configuration from YAML

Create `config/vectorstore.yaml`:

```yaml
name: production_vectorstore

embedding:
  model_type: sentence_transformer
  model_name: Snowflake/snowflake-arctic-embed-m
  batch_size: 128
  normalize: true

database:
  host: localhost
  port: 5432
  database: vectorstore_db
  user: postgres
  password: postgres

search:
  similarity_metric: cosine
  default_top_k: 5
  enable_hybrid_search: true
  hybrid_alpha: 0.7
  rerank: true

index:
  chunk_size: 512
  chunk_overlap: 50
  batch_insert_size: 10000

enable_caching: true
cache_size: 1000
enable_logging: true
log_level: INFO
```

Load and use:

```python
from rakam_system_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

# Load from YAML - all behavior controlled by config!
config = VectorStoreConfig.from_yaml("config/vectorstore.yaml")

# Or load from JSON
config = VectorStoreConfig.from_json("config/vectorstore.json")

# Or use the helper function (auto-detects format)
from rakam_system_vectorstore.config import load_config
config = load_config("config/vectorstore.yaml")

# Create store - behavior defined entirely by config
store = ConfigurablePgVectorStore(config=config)
store.setup()
```

**Example: Switching Embedding Models Without Code Changes**

```yaml
# config/vectorstore_local.yaml - Use local embeddings (free, private)
embedding:
  model_type: sentence_transformer
  model_name: Snowflake/snowflake-arctic-embed-m

# config/vectorstore_openai.yaml - Use OpenAI embeddings (better quality)
embedding:
  model_type: openai
  model_name: text-embedding-3-large

# Application code stays the same - just change config file!
# config = VectorStoreConfig.from_yaml("config/vectorstore_openai.yaml")
```

**Example: Enabling Hybrid Search Without Code Changes**

```yaml
# Before: Vector search only
search:
  similarity_metric: cosine
  default_top_k: 5
  enable_hybrid_search: false

# After: Hybrid search enabled - just update config!
search:
  similarity_metric: cosine
  default_top_k: 5
  enable_hybrid_search: true
  hybrid_alpha: 0.7
```

### Using Different Embedding Models

```python
from rakam_system_vectorstore.config import VectorStoreConfig, EmbeddingConfig

# Local embeddings with Sentence Transformers
config_local = VectorStoreConfig(
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",
        model_name="Snowflake/snowflake-arctic-embed-m",  # 768 dimensions
        # model_name="all-MiniLM-L6-v2",                  # 384 dimensions
        batch_size=128,
    )
)

# OpenAI embeddings
config_openai = VectorStoreConfig(
    embedding=EmbeddingConfig(
        model_type="openai",
        model_name="text-embedding-3-small",
        # api_key loaded from OPENAI_API_KEY env var
    )
)

# Cohere embeddings
config_cohere = VectorStoreConfig(
    embedding=EmbeddingConfig(
        model_type="cohere",
        model_name="embed-english-v3.0",
        # api_key loaded from COHERE_API_KEY env var
    )
)
```

### Multi-Model Support

Each embedding model automatically gets dedicated tables, preventing mixing of incompatible vector spaces:

```python
from rakam_system_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig, EmbeddingConfig

# Store using MiniLM model
config_minilm = VectorStoreConfig(
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",
        model_name="all-MiniLM-L6-v2"  # 384D
    )
)
store_minilm = ConfigurablePgVectorStore(config=config_minilm)
# Tables: application_nodeentry_all_minilm_l6_v2

# Store using Arctic model
config_arctic = VectorStoreConfig(
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",
        model_name="Snowflake/snowflake-arctic-embed-m"  # 768D
    )
)
store_arctic = ConfigurablePgVectorStore(config=config_arctic)
# Tables: application_nodeentry_snowflake_arctic_embed_m

# Both can coexist without conflicts!
```

> **Important**: Even if two models have the same dimensions, their vector spaces are different! Model-specific tables prevent meaningless search results from mixed embeddings.

### Hybrid Search

Combine vector similarity with keyword search:

```python
from rakam_system_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

config = VectorStoreConfig()
config.search.enable_hybrid_search = True
config.search.hybrid_alpha = 0.7  # 70% vector, 30% keyword

store = ConfigurablePgVectorStore(config=config)
store.setup()

# Regular search (uses config defaults)
results = store.search("machine learning algorithms", top_k=10)

# Hybrid search with custom alpha
results = store.hybrid_search(
    query="machine learning algorithms",
    top_k=10,
    alpha=0.5  # 50/50 split
)

for r in results:
    print(f"[{r['score']:.4f}] {r['content'][:100]}...")
```

### Keyword Search

Full-text search using PostgreSQL's BM25 or ts_rank:

```python
from rakam_system_vectorstore import ConfigurablePgVectorStore, VectorStoreConfig

# Configure keyword search
config = VectorStoreConfig(
    search={
        "keyword_ranking_algorithm": "bm25",  # or "ts_rank"
        "keyword_k1": 1.2,  # BM25 k1 parameter
        "keyword_b": 0.75   # BM25 b parameter
    }
)

store = ConfigurablePgVectorStore(config=config)
store.setup()

# Add some documents
from rakam_system_vectorstore import Node, NodeMetadata

nodes = [
    Node(
        content="Machine learning is a subset of artificial intelligence.",
        metadata=NodeMetadata(source_file_uuid="doc1", position=0)
    ),
    Node(
        content="Deep learning uses neural networks with multiple layers.",
        metadata=NodeMetadata(source_file_uuid="doc1", position=1)
    ),
    Node(
        content="Natural language processing helps computers understand text.",
        metadata=NodeMetadata(source_file_uuid="doc1", position=2)
    ),
]
store.add_nodes(nodes)

# Keyword search with BM25
results = store.keyword_search(
    query="machine learning neural networks",
    top_k=5,
    ranking_algorithm="bm25"
)

print("Keyword Search Results (BM25):")
for r in results:
    print(f"  [{r['score']:.4f}] {r['content'][:80]}...")

# Keyword search with ts_rank
results = store.keyword_search(
    query="artificial intelligence",
    top_k=5,
    ranking_algorithm="ts_rank"
)

print("\nKeyword Search Results (ts_rank):")
for r in results:
    print(f"  [{r['score']:.4f}] {r['content'][:80]}...")

store.shutdown()
```

**When to use:**

- **BM25**: Best for general text search, handles term frequency well
- **ts_rank**: Good for structured documents with varying importance
- **Hybrid Search**: Combines semantic (vector) + keyword for best results

### Full Example with All Features

```python
import os
import django
from django.conf import settings

# Configure Django
if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'rakam_system_vectorstore.components.vectorstore',
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.postgresql',
                'NAME': os.getenv('POSTGRES_DB', 'vectorstore_db'),
                'USER': os.getenv('POSTGRES_USER', 'postgres'),
                'PASSWORD': os.getenv('POSTGRES_PASSWORD', 'postgres'),
                'HOST': os.getenv('POSTGRES_HOST', 'localhost'),
                'PORT': os.getenv('POSTGRES_PORT', '5432'),
            }
        },
        DEFAULT_AUTO_FIELD='django.db.models.BigAutoField',
    )
    django.setup()

from rakam_system_vectorstore import (
    ConfigurablePgVectorStore,
    VectorStoreConfig,
    Node,
    NodeMetadata,
    VSFile
)
from rakam_system_vectorstore.components.loader import AdaptiveLoader

# Create configuration
config = VectorStoreConfig.from_yaml("config/vectorstore.yaml")

# Initialize store with context manager
with ConfigurablePgVectorStore(config=config) as store:

    # Load documents
    loader = AdaptiveLoader(config={
        "chunk_size": config.index.chunk_size,
        "chunk_overlap": config.index.chunk_overlap
    })

    # Load from file
    vsfile = loader.load_as_vsfile("documents/manual.pdf")
    store.add_vsfile(vsfile)

    # Or add nodes directly
    nodes = [
        Node(
            content="Python is excellent for data science and machine learning.",
            metadata=NodeMetadata(
                source_file_uuid="doc_001",
                position=0,
                custom={"category": "programming", "topic": "python"}
            )
        ),
        Node(
            content="PostgreSQL with pgvector enables efficient vector similarity search.",
            metadata=NodeMetadata(
                source_file_uuid="doc_001",
                position=1,
                custom={"category": "database", "topic": "vectors"}
            )
        ),
    ]
    store.add_nodes(nodes)

    # Search with filters
    results = store.search(
        query="vector database search",
        top_k=5,
        # Metadata filters can be applied here
    )

    print("Search Results:")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:80]}...")
        print(f"           Source: {r.get('source_file_uuid', 'N/A')}")

    # Hybrid search
    print("\nHybrid Search Results:")
    hybrid_results = store.hybrid_search(
        query="PostgreSQL vector",
        top_k=5,
        alpha=0.6
    )
    for r in hybrid_results:
        print(f"  [{r['score']:.4f}] {r['content'][:80]}...")

    # Get collection stats
    count = store.count()
    print(f"\nTotal vectors in store: {count}")

# Store automatically cleaned up
```

### Configuration Reference

#### EmbeddingConfig

| Option       | Type | Default                              | Description                                         |
| ------------ | ---- | ------------------------------------ | --------------------------------------------------- |
| `model_type` | str  | `sentence_transformer`               | Backend: `sentence_transformer`, `openai`, `cohere` |
| `model_name` | str  | `Snowflake/snowflake-arctic-embed-m` | Model identifier                                    |
| `api_key`    | str  | None                                 | API key (auto-loaded from env)                      |
| `batch_size` | int  | 128                                  | Batch size for encoding                             |
| `normalize`  | bool | True                                 | Normalize embeddings                                |
| `dimensions` | int  | None                                 | Vector dimensions (auto-detected)                   |

#### DatabaseConfig

| Option      | Type | Default          | Description                              |
| ----------- | ---- | ---------------- | ---------------------------------------- |
| `host`      | str  | `localhost`      | PostgreSQL host (or `POSTGRES_HOST` env) |
| `port`      | int  | 5432             | PostgreSQL port (or `POSTGRES_PORT` env) |
| `database`  | str  | `vectorstore_db` | Database name (or `POSTGRES_DB` env)     |
| `user`      | str  | `postgres`       | Username (or `POSTGRES_USER` env)        |
| `password`  | str  | `postgres`       | Password (or `POSTGRES_PASSWORD` env)    |
| `pool_size` | int  | 10               | Connection pool size                     |

#### SearchConfig

| Option                 | Type  | Default  | Description                           |
| ---------------------- | ----- | -------- | ------------------------------------- |
| `similarity_metric`    | str   | `cosine` | Metric: `cosine`, `l2`, `dot_product` |
| `default_top_k`        | int   | 5        | Default results count                 |
| `enable_hybrid_search` | bool  | True     | Enable keyword + vector search        |
| `hybrid_alpha`         | float | 0.7      | Vector weight (1-alpha for keyword)   |
| `rerank`               | bool  | True     | Rerank results                        |

#### IndexConfig

| Option              | Type | Default | Description            |
| ------------------- | ---- | ------- | ---------------------- |
| `chunk_size`        | int  | 512     | Chunk size in tokens   |
| `chunk_overlap`     | int  | 50      | Overlap between chunks |
| `batch_insert_size` | int  | 10000   | Batch size for inserts |

---

## Quick Start: RAG Pipeline

Build a complete Retrieval-Augmented Generation pipeline:

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_system_vectorstore.components.vectorstore.faiss_vector_store import FaissStore
from rakam_system_vectorstore.components.loader import AdaptiveLoader
from rakam_systems_core.interfaces.tool import ToolComponent

# Step 1: Load and index documents
loader = AdaptiveLoader(config={"chunk_size": 512, "chunk_overlap": 50})

# Load documents (supports PDF, DOCX, TXT, MD, HTML, etc.)
# nodes = loader.load_as_nodes("path/to/document.pdf")

# For demo, use sample text
from rakam_system_vectorstore.core import Node, NodeMetadata

sample_docs = [
    "Our company was founded in 2020 and specializes in AI solutions.",
    "We offer three main products: AI Assistant, Data Analytics, and Automation.",
    "The AI Assistant can handle customer queries 24/7 with 95% accuracy.",
    "Our pricing starts at $99/month for small businesses.",
    "Enterprise customers get dedicated support and custom integrations.",
    "We have offices in New York, London, and Tokyo.",
]

nodes = [
    Node(content=doc, metadata=NodeMetadata(source_file_uuid="company_info", position=i))
    for i, doc in enumerate(sample_docs)
]

# Step 2: Create vector store
store = FaissStore(
    name="rag_store",
    base_index_path="./rag_indexes",
    embedding_model="Snowflake/snowflake-arctic-embed-m",
    initialising=True
)
store.create_collection_from_nodes("knowledge_base", nodes)

# Step 3: Create search tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Search the knowledge base for relevant information."""
    results, _ = store.search(
        collection_name="knowledge_base",
        query=query,
        distance_type="cosine",
        number=top_k
    )

    context = "\n".join([
        f"- {content}" for _, (_, content, _) in results.items()
    ])
    return f"Relevant information:\n{context}"

search_tool = ToolComponent.from_function(
    function=search_knowledge_base,
    name="search_knowledge_base",
    description="Search the company knowledge base for information",
    json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "default": 3, "description": "Number of results"}
        },
        "required": ["query"],
        "additionalProperties": False
    }
)

# Step 4: Create RAG agent
async def main():
    agent = BaseAgent(
        name="rag_agent",
        model="openai:gpt-4o",
        system_prompt="""You are a helpful customer service agent for our company.
Use the search_knowledge_base tool to find relevant information before answering.
Always base your answers on the retrieved information.""",
        tools=[search_tool]
    )

    # Ask questions
    questions = [
        "When was the company founded?",
        "What products do you offer?",
        "How much does the service cost?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = await agent.arun(question)
        print(f"A: {result.output_text}")

asyncio.run(main())
```

---

## Quick Start: LLM Gateway

Use the LLM Gateway for direct LLM interactions:

```python
from rakam_systems_agent.components.llm_gateway import (
    OpenAIGateway,
    MistralGateway,
    LLMGatewayFactory,
    LLMRequest
)

# Create gateway using factory
gateway = LLMGatewayFactory.create(
    provider="openai",
    model="gpt-4o"
)

# Simple text generation
request = LLMRequest(
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain quantum computing in simple terms.",
    temperature=0.7,
    max_tokens=200
)

response = gateway.generate(request)
print(f"Response: {response.content}")
print(f"Tokens used: {response.usage}")
```

### Structured Output with Gateway

```python
from pydantic import BaseModel
from rakam_systems_agent.components.llm_gateway import OpenAIGateway, LLMRequest

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    instructions: list[str]
    prep_time_minutes: int
    servings: int

gateway = OpenAIGateway(model="gpt-4o")

request = LLMRequest(
    system_prompt="You are a chef. Create recipes based on user requests.",
    user_prompt="Give me a simple pasta recipe."
)

recipe = gateway.generate_structured(request, Recipe)
print(f"Recipe: {recipe.name}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")
print(f"Ingredients: {', '.join(recipe.ingredients)}")
```

### Streaming with Gateway

```python
from rakam_systems_agent.components.llm_gateway import OpenAIGateway, LLMRequest

gateway = OpenAIGateway(model="gpt-4o")

request = LLMRequest(
    user_prompt="Write a poem about coding.",
    temperature=0.8
)

print("Poem:\n")
for chunk in gateway.stream(request):
    print(chunk, end="", flush=True)
print()
```

---

## Quick Start: Document Loading

Load and process various document types:

```python
from rakam_system_vectorstore.components.loader import AdaptiveLoader

# Create loader
loader = AdaptiveLoader(config={
    "chunk_size": 512,
    "chunk_overlap": 50,
    "encoding": "utf-8"
})

# Load as text
text = loader.load_as_text("document.pdf")
print(f"Loaded {len(text)} characters")

# Load as chunks
chunks = loader.load_as_chunks("document.pdf")
print(f"Created {len(chunks)} chunks")

# Load as nodes (with metadata)
nodes = loader.load_as_nodes(
    "document.pdf",
    custom_metadata={"category": "technical", "author": "John"}
)
print(f"Created {len(nodes)} nodes")

# Load as VSFile (complete document representation)
vsfile = loader.load_as_vsfile("document.pdf")
print(f"File: {vsfile.file_name}, UUID: {vsfile.uuid}")
```

### Supported File Types

| Type             | Extensions                                                |
| ---------------- | --------------------------------------------------------- |
| **Text**         | `.txt`, `.text`                                           |
| **Markdown**     | `.md`, `.markdown`                                        |
| **PDF**          | `.pdf`                                                    |
| **Word**         | `.docx`, `.doc`                                           |
| **OpenDocument** | `.odt`                                                    |
| **HTML**         | `.html`, `.htm`, `.xhtml`                                 |
| **Email**        | `.eml`, `.msg`                                            |
| **Data**         | `.json`, `.csv`, `.tsv`, `.xlsx`, `.xls`                  |
| **Code**         | `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs`, `.rb` |

---

## Quick Start: Configuration from YAML

Load agents and tools from configuration files:

```yaml
# config/agents.yaml
version: "1.0"

prompts:
  assistant:
    system_prompt: |
      You are a helpful AI assistant.
      Always be accurate and concise.

tools:
  get_time:
    name: "get_time"
    type: "direct"
    module: "datetime"
    function: "datetime.now"
    description: "Get current date and time"
    json_schema:
      type: "object"
      properties: {}

agents:
  main_agent:
    name: "main_agent"
    model_config:
      model: "openai:gpt-4o"
      temperature: 0.7
    prompt_config: "assistant"
    tools:
      - "get_time"
    enable_tracking: true
```

```python
from rakam_systems_core.config_loader import ConfigurationLoader

loader = ConfigurationLoader()

# Load configuration
config = loader.load_from_yaml("config/agents.yaml")

# Create agent from config
agent = loader.create_agent("main_agent", config)

# Use the agent
result = await agent.arun("What time is it?")
```

---

## Next Steps

Now that you've completed the quick start:

1. **Read the Full Documentation**
   - [Components Guide](components.md) - Detailed component documentation
   - [Development Guide](development_guide.md) - How to extend the framework
   - [Installation Guide](installation.md) - Advanced installation options

2. **Explore More Examples**
   - `rakam_systems/examples/ai_agents_examples/` - Agent examples
   - `rakam_systems/examples/ai_vectorstore_examples/` - Vector store examples
   - `rakam_systems/examples/configs/` - Configuration examples

3. **Build Your Application**
   - Start with a simple agent
   - Add tools for your specific use case
   - Integrate vector storage for RAG
   - Scale with PostgreSQL for production

4. **Join the Community**
   - [GitHub Repository](https://github.com/Rakam-AI/rakam_systems)
   - Report issues and request features
   - Contribute to the project

---

## Common Patterns

### Error Handling

```python
import asyncio
from rakam_systems_agent import BaseAgent

async def main():
    agent = BaseAgent(
        name="safe_agent",
        model="openai:gpt-4o",
        system_prompt="You are helpful."
    )

    try:
        result = await agent.arun("Hello!")
        print(result.output_text)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
```

### Context Manager Pattern

```python
from rakam_system_vectorstore import ConfigurablePgVectorStore

# Automatic setup and cleanup
with ConfigurablePgVectorStore(config=config) as store:
    store.add_nodes(nodes)
    results = store.search("query")
# shutdown() called automatically
```

### Async Batch Processing

```python
import asyncio
from rakam_systems_agent import BaseAgent

async def process_queries(queries: list[str]):
    agent = BaseAgent(
        name="batch_agent",
        model="openai:gpt-4o",
        system_prompt="Answer concisely."
    )

    # Process queries concurrently
    tasks = [agent.arun(q) for q in queries]
    results = await asyncio.gather(*tasks)

    return [r.output_text for r in results]

queries = ["What is Python?", "What is JavaScript?", "What is Rust?"]
answers = asyncio.run(process_queries(queries))
for q, a in zip(queries, answers):
    print(f"Q: {q}\nA: {a}\n")
```

---

**Happy Building! 🚀**

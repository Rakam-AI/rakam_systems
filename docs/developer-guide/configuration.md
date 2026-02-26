---
title: Configuration System
---

# Configuration System

**The Core Advantage: Configuration Without Code Changes**

Rakam Systems embraces a configuration-first approach, allowing you to modify agent behavior, vector store settings, and system parameters without touching your application code.

## Benefits of Configuration-First Design

1. **Rapid Iteration**: Test different models, prompts, or parameters instantly
2. **Environment Management**: Use different configs for dev/staging/production
3. **A/B Testing**: Compare performance of different settings by swapping configs
4. **Team Collaboration**: Non-developers can tune prompts and parameters
5. **Cost Optimization**: Switch to cheaper models for development, expensive for production
6. **Risk Reduction**: Change behavior without code deployment risks

## Real-World Scenarios

**Scenario 1: Model Optimization**

```yaml
# Week 1: Start with GPT-4o
model: openai:gpt-4o
temperature: 0.7

# Week 2: Test GPT-4o-mini for cost savings (no code changes!)
model: openai:gpt-4o-mini
temperature: 0.7

# Week 3: Back to GPT-4o for production (just revert config)
model: openai:gpt-4o
temperature: 0.5  # Also tuned temperature
```

**Scenario 2: Search Algorithm Testing**

```yaml
# Current: Using BM25 for keyword search
search:
  keyword_ranking_algorithm: bm25

# Test: Try ts_rank (just update config, no code changes!)
search:
  keyword_ranking_algorithm: ts_rank

# Decide: Compare results and keep the best one
```

**Scenario 3: Multi-Environment Deployment**

```python
# Application code (never changes)
import os
config_file = os.getenv("AGENT_CONFIG", "config/agent_config.yaml")
config = loader.load_from_yaml(config_file)
agent = loader.create_agent("my_agent", config)

# Dev: AGENT_CONFIG=config/agent_config_dev.yaml
# Staging: AGENT_CONFIG=config/agent_config_staging.yaml
# Prod: AGENT_CONFIG=config/agent_config_prod.yaml
```

## VectorStoreConfig

```python
from rakam_systems_vectorstore.config import (
    VectorStoreConfig,
    EmbeddingConfig,
    DatabaseConfig,
    SearchConfig,
    IndexConfig,
    load_config
)

# Programmatic configuration
config = VectorStoreConfig(
    name="my_vectorstore",
    embedding=EmbeddingConfig(
        model_type="sentence_transformer",
        model_name="Snowflake/snowflake-arctic-embed-m",
        batch_size=128,
        normalize=True
    ),
    database=DatabaseConfig(
        host="localhost",
        port=5432,
        database="vectorstore_db",
        user="postgres",
        password="postgres"
    ),
    search=SearchConfig(
        similarity_metric="cosine",
        default_top_k=5,
        enable_hybrid_search=True,
        hybrid_alpha=0.7
    ),
    index=IndexConfig(
        chunk_size=512,
        chunk_overlap=50,
        batch_insert_size=10000
    )
)

# From YAML file
config = VectorStoreConfig.from_yaml("config.yaml")

# From JSON file
config = VectorStoreConfig.from_json("config.json")

# From dictionary
config = VectorStoreConfig.from_dict(config_dict)

# Validation
config.validate()

# Save configuration
config.save_yaml("output_config.yaml")
config.save_json("output_config.json")
```

## YAML Configuration Example

```yaml
# vectorstore_config.yaml
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

index:
  chunk_size: 512
  chunk_overlap: 50
  batch_insert_size: 10000

enable_caching: true
cache_size: 1000
enable_logging: true
log_level: INFO
```

## Agent Configuration Example

```yaml
# agent_config.yaml
agents:
  my_agent:
    name: my_agent
    llm_config:
      model: openai:gpt-4o
      temperature: 0.7
      max_tokens: 2000
      parallel_tool_calls: true
    prompt_config: default_prompt
    tools:
      - search_tool
      - calculator
    deps_type: myapp.models.AgentDeps
    output_type:
      name: AgentOutput
      fields:
        answer:
          type: str
          description: The answer
        confidence:
          type: float
          description: Confidence score
    enable_tracking: true

prompts:
  default_prompt:
    system_prompt: |
      You are a helpful AI assistant.
      Always provide accurate and helpful responses.

tools:
  search_tool:
    name: search_tool
    type: direct
    module: myapp.tools
    function: search
    description: Search for information
    json_schema:
      type: object
      properties:
        query:
          type: string
      required: [query]

  calculator:
    name: calculator
    type: direct
    module: myapp.tools
    function: calculate
    description: Perform calculations
```

## Quick Start Examples

### Basic Agent

```python
import asyncio
from rakam_systems_agent import BaseAgent

async def main():
    agent = BaseAgent(
        name="assistant",
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant."
    )

    result = await agent.arun("What is the capital of France?")
    print(result.output_text)

asyncio.run(main())
```

### Agent with Tools

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_core.interfaces.tool import ToolComponent

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny, 25°C"

weather_tool = ToolComponent.from_function(
    function=get_weather,
    name="get_weather",
    description="Get the current weather for a city",
    json_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
)

async def main():
    agent = BaseAgent(
        name="weather_assistant",
        model="openai:gpt-4o",
        system_prompt="You help users with weather information.",
        tools=[weather_tool]
    )

    result = await agent.arun("What's the weather in Paris?")
    print(result.output_text)

asyncio.run(main())
```

### Document Search Pipeline

```python
from rakam_systems_vectorstore import (
    ConfigurablePgVectorStore,
    VectorStoreConfig,
    AdaptiveLoader
)

# Configure vector store
config = VectorStoreConfig()
store = ConfigurablePgVectorStore(config=config)
store.setup()

# Load documents
loader = AdaptiveLoader(config={"chunk_size": 512})
nodes = loader.load_as_nodes("documents/research_paper.pdf")

# Add to vector store
store.add_nodes(nodes)

# Search
results = store.search("What are the main findings?", top_k=5)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:200]}...")
    print("---")

store.shutdown()
```

### Full RAG Pipeline

```python
import asyncio
from rakam_systems_agent import BaseAgent
from rakam_systems_vectorstore import ConfigurablePgVectorStore, AdaptiveLoader, VectorStoreConfig
from rakam_systems_core.interfaces.tool import ToolComponent

# Setup vector store
config = VectorStoreConfig()
store = ConfigurablePgVectorStore(config=config)
store.setup()

# Index documents
loader = AdaptiveLoader()
for doc_path in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
    nodes = loader.load_as_nodes(doc_path)
    store.add_nodes(nodes)

# Create search tool
def search_documents(query: str, top_k: int = 5) -> str:
    results = store.search(query, top_k=top_k)
    return "\n\n".join([r['content'] for r in results])

search_tool = ToolComponent.from_function(
    function=search_documents,
    name="search_documents",
    description="Search the document database",
    json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results"}
        },
        "required": ["query"]
    }
)

# Create RAG agent
async def main():
    agent = BaseAgent(
        name="rag_agent",
        model="openai:gpt-4o",
        system_prompt="""You are a helpful assistant with access to a document database.
        Use the search_documents tool to find relevant information before answering questions.""",
        tools=[search_tool]
    )

    result = await agent.arun("What are the key points from the documents?")
    print(result.output_text)

asyncio.run(main())
store.shutdown()
```

## Logging Utilities

The core package includes logging utilities:

```python
from rakam_systems_tools.utils import logging

logger = logging.getLogger(__name__)
logger.info("Processing document...")
logger.debug("Detailed debug info")
logger.error("An error occurred")
```

## Environment Variables

The system supports the following environment variables:

| Variable            | Description         | Used By                               |
| ------------------- | ------------------- | ------------------------------------- |
| `OPENAI_API_KEY`    | OpenAI API key      | OpenAIGateway, ConfigurableEmbeddings |
| `MISTRAL_API_KEY`   | Mistral API key     | MistralGateway                        |
| `COHERE_API_KEY`    | Cohere API key      | ConfigurableEmbeddings                |
| `POSTGRES_HOST`     | PostgreSQL host     | DatabaseConfig                        |
| `POSTGRES_PORT`     | PostgreSQL port     | DatabaseConfig                        |
| `POSTGRES_DB`       | PostgreSQL database | DatabaseConfig                        |
| `POSTGRES_USER`     | PostgreSQL user     | DatabaseConfig                        |
| `POSTGRES_PASSWORD` | PostgreSQL password | DatabaseConfig                        |

## Best Practices

1. **Use context managers** or explicit `setup()`/`shutdown()` for proper resource management
2. **Prefer configuration files** over hardcoded values for production deployments
3. **Enable tracking** during development to support debugging and evaluation
4. **Keep model-specific tables** (the default) to prevent mixing incompatible vector spaces
5. **Batch operations** when processing large document collections
6. **Prefer async methods** (`arun`, `astream`) for agents — they are powered by Pydantic AI
7. **Validate configurations** before deployment with `config.validate()` or `loader.validate_config()`

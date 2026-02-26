---
title: Configure with YAML
---

# Configure with YAML

## Agent configuration

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

Load and use agent configurations:

```python
from rakam_systems_core.config_loader import ConfigurationLoader

loader = ConfigurationLoader()
config = loader.load_from_yaml("agent_config.yaml")

# Create agents from config
agent = loader.create_agent("my_agent", config)
all_agents = loader.create_all_agents(config)

# Get tool registry
registry = loader.get_tool_registry(config)

# Validate configuration
is_valid, errors = loader.validate_config("config.yaml")
```

## VectorStore configuration

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

Load programmatically:

```python
from rakam_systems_vectorstore.config import (
    VectorStoreConfig,
    EmbeddingConfig,
    DatabaseConfig,
    SearchConfig,
    IndexConfig,
    load_config
)

# From YAML file
config = VectorStoreConfig.from_yaml("config.yaml")

# From JSON file
config = VectorStoreConfig.from_json("config.json")

# From dictionary
config = VectorStoreConfig.from_dict(config_dict)

# Programmatic
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

# Validate
config.validate()

# Save
config.save_yaml("output_config.yaml")
config.save_json("output_config.json")
```

## Multi-environment deployment

Use environment variables to select configuration files without changing application code:

```python
import os
config_file = os.getenv("AGENT_CONFIG", "config/agent_config.yaml")
config = loader.load_from_yaml(config_file)
agent = loader.create_agent("my_agent", config)

# Dev: AGENT_CONFIG=config/agent_config_dev.yaml
# Staging: AGENT_CONFIG=config/agent_config_staging.yaml
# Prod: AGENT_CONFIG=config/agent_config_prod.yaml
```

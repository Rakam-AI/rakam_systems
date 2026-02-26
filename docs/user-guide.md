---
title: User Guide
---

import Prerequisites from './_partials/_prerequisites.md';

This guide is for AI engineers using Rakam Systems to build AI systems. It covers installation, environment setup, agents, vector stores, evaluation, and cloud storage.

<Prerequisites />

## Installation

### Install all packages

Install all Rakam Systems packages at once:

```bash
pip install rakam-systems
```

This is the recommended starting point. It includes the core, agent, and vectorstore packages.

### Install specific packages

Rakam Systems uses a modular architecture. If you want to reduce dependencies, you can install only what you need:

```bash
# Core only (required by all other packages)
pip install rakam-systems-core

# Agent package (includes core)
pip install rakam-systems-agent[all]

# Vectorstore package (includes core)
pip install rakam-systems-vectorstore

# Agent + Vectorstore (for RAG applications)
pip install rakam-systems-agent[all] rakam-systems-vectorstore
```

### Review dependencies

| Package | Purpose | Min specs | Key dependencies |
|---------|---------|-----------|-----------------|
| `rakam-systems-core` | Foundational interfaces and utilities. Required by all other packages. | — | `pydantic`, `pyyaml`, `python-dotenv`, … |
| `rakam-systems-agent` | AI agent implementations powered by Pydantic AI. | 4 GB RAM, internet access | Core + `pydantic-ai`, `mistralai`, `openai`, … |
| `rakam-systems-vectorstore` | Vector storage and document processing. Requires PostgreSQL + pgvector for persistent storage. | 8 GB+ RAM, 5 GB+ disk | Core + `sentence-transformers`, `faiss-cpu`, `torch`, … |


## Environment Setup

### Configure API keys

Create a `.env` file in your project root. Not all keys are required — they depend on which providers and features you use.

```bash
# OpenAI — required for GPT models and OpenAI embeddings
OPENAI_API_KEY=sk-your-openai-key

# Mistral AI — required for Mistral models
MISTRAL_API_KEY=your-mistral-key

# Cohere — required for Cohere embeddings
COHERE_API_KEY=your-cohere-key

# HuggingFace — required for private or gated models
HUGGINGFACE_TOKEN=your-hf-token
```

Load in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Configure PostgreSQL

PostgreSQL with pgvector is required if you use `ConfigurablePgVectorStore` or PostgreSQL-backed chat history. This guide assumes a PostgreSQL instance with pgvector is already available.

Configure the connection via environment variables:

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vectorstore_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```


## Building AI Systems

This section covers the main components for building applications with Rakam Systems: agents, vector stores, RAG pipelines, and YAML-based configuration.

### Build agents

The following examples cover the main usage patterns of `BaseAgent`.

#### Run a basic agent

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

#### Stream responses

```python
async def main():
    agent = BaseAgent(name="stream_agent", model="openai:gpt-4o")

    print("Response: ", end="", flush=True)
    async for chunk in agent.astream("Tell me a short story."):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

#### Customize model settings

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

#### Add tools

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

#### Use structured output

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

### Use the vector store

Vector Store is a standalone component for storing and searching document embeddings. It can be used independently for semantic search, or as the data layer in a RAG pipeline (see below).

#### Use FAISS (in-memory)

```python
from rakam_systems_vectorstore import FaissStore, Node, NodeMetadata

store = FaissStore(
    name="my_store",
    base_index_path="./indexes",
    embedding_model="Snowflake/snowflake-arctic-embed-m",
    initialising=True
)

nodes = [
    Node(content="Python is a programming language.",
         metadata=NodeMetadata(source_file_uuid="doc1", position=0)),
    Node(content="Machine learning is AI subset.",
         metadata=NodeMetadata(source_file_uuid="doc1", position=1)),
]
store.create_collection_from_nodes("docs", nodes)

results, _ = store.search(collection_name="docs", query="What is Python?", number=3)
for _, (_, content, dist) in results.items():
    print(f"[{dist:.4f}] {content}")
```

#### Use PostgreSQL (production)

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

### Build a RAG pipeline

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


### Configure an agent with YAML

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


## Evaluation

The evaluation service must be running to use evaluation features. Contact us if you need help setting it up.

Configure access to the evaluation service in your `.env`:

```bash
# Evaluation service access
EVALFRAMEWORK_URL="http://eval-service-url.com"   # URL of the evaluation service
EVALFRAMEWORK_API_KEY="your-api-token"             # Generate from the /docs Swagger UI
```

### Write an evaluation function

Create an `eval/` directory in your project and add evaluation functions decorated with `@eval_run`. Each function returns an `EvalConfig` or `SchemaEvalConfig`.

#### Text evaluation

```python
# eval/examples.py
from rakam_systems_cli.decorators import eval_run
from rakam_systems_tools.evaluation.schema import (
    EvalConfig,
    TextInputItem,
    ClientSideMetricConfig,
    ToxicityConfig,
    CorrectnessConfig,
)

@eval_run
def test_simple_text_eval():
    """A simple text evaluation showcasing a basic client-side metric."""
    return EvalConfig(
        component="text_component_1",
        label="demo_simple_text",
        data=[
            TextInputItem(
                id="txt_001",
                input="Hello world",
                output="Hello world",
                expected_output="Hello world",
                metrics=[ClientSideMetricConfig(name="relevance", score=1)],
            )
        ],
        metrics=[ToxicityConfig(name="toxicity_demo", include_reason=False)],
    )
```

Available text metrics: `CorrectnessConfig`, `AnswerRelevancyConfig`, `FaithfulnessConfig`, `ToxicityConfig`.

#### Schema evaluation

```python
from rakam_systems_cli.decorators import eval_run
from rakam_systems_tools.evaluation.schema import (
    SchemaEvalConfig,
    SchemaInputItem,
    JsonCorrectnessConfig,
)

@eval_run
def test_json_output():
    """Validate JSON structure of model outputs."""
    return SchemaEvalConfig(
        component="json-generator",
        label="json_validation",
        data=[
            SchemaInputItem(
                input="Generate a JSON object with name and age.",
                output='{"name": "John", "age": 30}'
            )
        ],
        metrics=[
            JsonCorrectnessConfig(
                excpected_schema={"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
            )
        ],
    )
```

> **Note:** The parameter name `excpected_schema` is misspelled in the SDK. Use it as shown above — this is a known upstream issue.

Available schema metrics: `JsonCorrectnessConfig`, `FieldsPresenceConfig`.

#### Client-side metrics

Log metrics calculated in your own code. These are sent alongside input data without server-side evaluation:

```python
TextInputItem(
    input="User review",
    output="I am happy with this product.",
    metrics=[
        ClientSideMetricConfig(
            name="sentiment",
            score=1.0,
            reason="The user expressed a positive sentiment."
        )
    ]
)
```

Pass an empty list to `metrics` in `EvalConfig` to skip server-side evaluation.

#### Probabilistic evaluation

Use `maybe_*` methods to run evaluations on a sample of requests, reducing load on the evaluation service:

```python
from rakam_systems_tools.evaluation import DeepEvalClient

client = DeepEvalClient()

# Runs approximately 10% of the time
client.maybe_text_eval(data=data, metrics=metrics, chance=0.1)
```

#### Error handling

By default, the evaluation client returns a dictionary with an `"error"` key on failure. Set `raise_exception=True` to raise instead:

```python
from rakam_systems_tools.evaluation import DeepEvalClient

client = DeepEvalClient()

try:
    result = client.text_eval(data=data, metrics=metrics, raise_exception=True)
except requests.RequestException as e:
    print(f"An error occurred: {e}")
```

### Run evaluations

Install the CLI package:

```bash
pip install rakam-systems-cli
```

#### Execute evaluations

The `run` command discovers and executes all `@eval_run`-decorated functions in the target directory:

```bash
# Run all evaluations in ./eval (default)
rakam eval run

# Run from a different directory
rakam eval run path/to/evals

# Search subdirectories recursively
rakam eval run --recursive

# Preview which functions would run without executing them
rakam eval run --dry-run

# Save each run result to a local JSON file
rakam eval run --save-runs --output-dir ./eval_runs
```

Example dry-run output:

```
📄 eval/quality.py
  ▶ test_answer_relevance
    🧪 Dry-run OK → text_eval
  ▶ test_json_output
    🧪 Dry-run OK → schema_eval

📄 eval/safety.py
  ▶ test_toxicity
    🧪 Dry-run OK → text_eval
```

#### View results

Show the details of a specific run, or the most recent one by default:

```bash
# Show the most recent run
rakam eval show

# Show a specific run by ID
rakam eval show --id 42

# Show a run by tag
rakam eval show --tag baseline-v1

# Output raw JSON (useful for scripting)
rakam eval show --raw
```

#### Compare runs

Compare two evaluation runs to track quality changes between iterations. Provide exactly two targets using `--id` or `--tag`:

```bash
# Compare two runs by ID
rakam eval compare --id 42 --id 45

# Compare a run by ID with a tagged run
rakam eval compare --id 42 --tag baseline-v1

# Show a summary diff only (reduced output)
rakam eval compare --id 42 --id 45 --summary

# Show a side-by-side diff
rakam eval compare --id 42 --id 45 --side-by-side
```

Example summary output:

```
Summary:
  | Status       | # | Metrics                |
  |--------------|---|------------------------|
  | ↑ Improved   | 2 | relevance, correctness |
  | ↓ Regressed  | 1 | faithfulness           |
  | ± Unchanged  | 1 | toxicity               |
  | + Added.     | 0 | -                      |
  | - Removed.   | 0 | -                      |
```

The default compare mode produces a unified diff of the full run payloads. Use `--summary` for a quick overview of what improved or regressed.

#### Tag runs

Assign human-readable tags to runs for easier reference in `show` and `compare`:

```bash
# Assign a tag to a run
rakam eval tag --id 42 --tag baseline-v1

# Delete a tag
rakam eval tag --delete baseline-v1
```

```
✅ Tag assigned successfully
Run ID: 42
Tag: baseline-v1
```

Tags let you compare named checkpoints (e.g., `--tag baseline-v1 --tag after-prompt-update`) instead of remembering numeric IDs.

#### List runs and evaluations

```bash
# List recent runs (newest first, default 20)
rakam eval list runs

# List more runs
rakam eval list runs --limit 50

# List all @eval_run functions discovered in ./eval
rakam eval list evals

# List all metric types used across evaluation functions
rakam eval metrics list
```

Example `list runs` output:

```
[id] tag                 label               created_at
[45] after-prompt-update demo_simple_text     2025-01-15 14:32:10
[44] -                   json_validation      2025-01-15 14:30:05
[42] baseline-v1         demo_simple_text     2025-01-14 09:15:22
[41] -                   toxicity_check       2025-01-14 09:12:00
```

## Cloud storage (S3)

The `rakam-system-tools` package includes a lightweight wrapper around boto3 for S3-compatible storage (AWS S3, OVH, Scaleway, MinIO).

```bash
pip install rakam-system-tools
```

### Configure S3

Set these environment variables in your `.env` file:

```bash
# Required
S3_ACCESS_KEY=your_access_key_here
S3_SECRET_KEY=your_secret_key_here
S3_BUCKET_NAME=your-bucket-name

# Optional (for S3-compatible services like OVH, Scaleway, MinIO)
S3_ENDPOINT_URL=https://s3.gra.io.cloud.ovh.net
S3_REGION=gra
```

For AWS S3, omit `S3_ENDPOINT_URL`. For other providers, set it to their endpoint (e.g., `https://s3.fr-par.scw.cloud` for Scaleway, `http://localhost:9000` for MinIO).

### Use S3

```python
from rakam_systems_tools.utils import s3

# Upload
s3.upload_file(
    key="documents/report.txt",
    content="Hello World!",
    content_type="text/plain"
)

# Download (returns bytes; decode for text)
content = s3.download_file("documents/report.txt")
print(content.decode("utf-8"))

# Check existence
if s3.file_exists("documents/report.txt"):
    print("File exists!")

# List files with prefix
files = s3.list_files(prefix="documents/")
for file in files:
    print(f"{file['Key']} - {file['Size']} bytes")

# Delete
s3.delete_file("documents/report.txt")
```

### Handle S3 errors

```python
from rakam_systems_tools.utils import s3

try:
    content = s3.download_file("missing-file.txt")
except s3.S3NotFoundError:
    print("File not found")
except s3.S3PermissionError:
    print("Access denied")
except s3.S3ConfigError:
    print("Configuration error")
except s3.S3Error as e:
    print(f"General S3 error: {e}")
```

Exception hierarchy: `S3Error` → `S3ConfigError`, `S3NotFoundError`, `S3PermissionError`.

## Troubleshooting

### ModuleNotFoundError

```
ModuleNotFoundError: No module named 'rakam_systems'
```

Verify the package is installed:

```bash
pip install rakam-systems
```

### Missing optional dependencies

```
ImportError: cannot import name 'BaseAgent' from 'rakam_systems_agent'
```

Install the required package:

```bash
pip install rakam-systems-agent
```

### Django not configured

```
django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS...
```

Configure Django before importing Django-dependent components:

```python
import django
from django.conf import settings

settings.configure(
    INSTALLED_APPS=['rakam_systems_vectorstore.components.vectorstore'],
    DATABASES={'default': {...}}
)
django.setup()

from rakam_systems_vectorstore import ConfigurablePgVectorStore
```

### PyTorch installation issues

PyTorch is large (~2 GB). For CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### FAISS GPU support

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

#### libmagic not found

On macOS:

```bash
brew install libmagic
```

On Ubuntu/Debian:

```bash
apt-get install libmagic1
```

On Windows:

```bash
pip install python-magic-bin
```

#### PostgreSQL connection refused

```bash
# Check if running
docker ps | grep postgres

# Start if not running
docker start postgres-vectorstore

# Or start with docker compose
docker compose up -d postgres
```

### Verify your installation

```python
# Core
from rakam_systems_core.base import BaseComponent
from rakam_systems_core.interfaces import ToolComponent, VectorStore
print("✅ Core installed successfully!")
```

```python
# Agent
try:
    from rakam_systems_agent import BaseAgent
    from rakam_systems_core.interfaces.agent import AgentInput, AgentOutput
    print("✅ AI Agents installed successfully!")
except ImportError as e:
    print(f"❌ AI Agents not installed: {e}")
```

```python
# Vectorstore
try:
    from rakam_systems_vectorstore import VectorStoreConfig, Node, VSFile, ConfigurableEmbeddings
    print("✅ AI Vectorstore installed successfully!")
except ImportError as e:
    print(f"❌ AI Vectorstore not installed: {e}")
```

```python
# LLM Gateway
try:
    from rakam_systems_agent.components.llm_gateway import OpenAIGateway, MistralGateway, LLMGatewayFactory
    print("✅ LLM Gateway installed successfully!")
except ImportError as e:
    print(f"❌ LLM Gateway not installed: {e}")
```

### Get help

- **Issues:** [GitHub Issues](https://github.com/Rakam-AI/rakam_systems/issues)

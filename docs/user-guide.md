---
title: User Guide
---

## Installation

### Option 1: Install All Packages

```bash
# Install all  components
pip install  rakam-systems==0.2.5rc9
```

### Option 2: Install Specific Packages

```bash
# Core only (required by others)
pip install  rakam-systems-core

# Agent package (includes core as dependency)
pip install rakam-systems-agent[all]==0.1.1rc12

# Vectorstore package (includes core as dependency)
pip install rakam-systems-agent[all]==0.1.1rc11

# Agent + Vectorstore (for RAG applications)
pip install rakam-systems-agent[all]==0.1.1rc12 rakam-systems-agent[all]==0.1.1rc11
```

### Packages Deps

Rakam Systems uses a modular architecture with three independent packages. Install only what you need:

#### Core Package (Required)

The core package provides foundational interfaces and utilities. It's required by both agent and vectorstore packages.

```bash
pip install rakam-systems-core
```

**Includes:**

- `pydantic` - Data validation
- `pyyaml` - YAML configuration
- `python-dotenv` - Environment variables
- `colorlog` - Logging
- `requests` - HTTP client

#### Agent Package

The agent package provides AI agent implementations powered by Pydantic AI.

```bash
pip install rakam-systems-agent
```

**Minimum Specs:**

- **RAM**: 4 GB minimum
- **Network**: Internet access for LLM API calls
- **API Keys**: OpenAI and/or Mistral API keys

**Includes:**

- All core package dependencies
- `pydantic-ai` - Agent framework
- `mistralai` - Mistral AI provider
- `openai` - OpenAI provider
- `tiktoken` - Token counting

**Use Cases:**

- Building AI agents with tools
- Multi-step reasoning systems
- Conversational AI
- Structured output generation

#### Vectorstore Package

The vectorstore package provides vector database solutions and document processing.

```bash
pip install rakam-systems-vectorstore
```

**Minimum Specs:**

- **RAM**: 8 GB+ (embedding models are loaded into memory)
- **Disk**: 5 GB+ (for downloading model weights)
- **GPU**: Optional but recommended for faster inference
- **PostgreSQL**: 12+ with pgvector extension (for persistent storage)

**Includes:**

| Package                 | Purpose                          |
| ----------------------- | -------------------------------- |
| `sentence-transformers` | Local embedding models           |
| `huggingface-hub`       | HuggingFace model authentication |
| `faiss-cpu`             | Vector similarity search         |
| `psycopg2-binary`       | PostgreSQL driver                |
| `pgvector`              | PostgreSQL vector extension      |
| `django`                | ORM for database operations      |
| `torch`                 | Deep learning backend            |
| `pymupdf`               | PDF processing                   |
| `pymupdf4llm`           | Lightweight PDF to markdown      |
| `python-docx`           | DOCX processing                  |
| `beautifulsoup4`        | HTML parsing                     |
| `chonkie`               | Text chunking                    |
| `docling`               | Advanced document processing     |
| `openai`                | OpenAI embeddings (optional)     |
| `cohere`                | Cohere embeddings (optional)     |
| `odfpy`                 | ODT file processing              |
| `openpyxl`              | Excel file processing            |
| `pandas`                | Tabular data handling            |
| `joblib`                | Parallel processing              |

**Use Cases:**

- Semantic search applications
- RAG (Retrieval-Augmented Generation)
- Document Q&A systems
- Knowledge base management

---

## Environment Setup

### Start the Evaluation Service (required for evaluation)

To launch all required backend services, run the following command (replace `YOUR_API` with your actual OpenAI API key):

```bash
docker run -d \
  --name eval-framework \
  -p 8080:8000 \
  -e OPENAI_API_KEY=YOUR_API \
  -e API_PREFIX="/eval-framework" \
  -e APP_NAME="eval-framework" \
  346k0827.c1.de1.container-registry.ovh.net/monitoring/evaluation-service:v0.2.4rc8
```

### API Keys

Create a `.env` file in your project root:

```bash
# OpenAI (for GPT models and embeddings)
OPENAI_API_KEY=sk-your-openai-key

# Mistral AI (for Mistral models)
MISTRAL_API_KEY=your-mistral-key

# Cohere (for Cohere embeddings)
COHERE_API_KEY=your-cohere-key

# HuggingFace (for private/gated models)
HUGGINGFACE_TOKEN=your-hf-token


# These will come from previous step
EVALFRAMEWORK_URL="http://eval-service-url.com" # url of docker container
EVALFRAMEWORK_API_KEY="your-api-token" # can be generated from '/docs' swagger-ui
```

Load in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

### PostgreSQL with pgvector

#### Option 1: Docker (Recommended)

```bash
docker run -d \
  --name postgres-vectorstore \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vectorstore_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

#### Option 2: Docker Compose

Create `docker-compose.yml`:

```yaml
version: "3.8"
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: vectorstore_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Run:

```bash
docker compose up -d
```

#### Environment Variables for PostgreSQL

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=vectorstore_db
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
```

### Django Configuration

For PostgreSQL-backed vector stores, configure Django:

```python
import os
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'rakam_systems_vectorstore.components.vectorstore',
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
```

---

## Agents

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

## Vector Store

### FAISS (in-memory, for development)

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

### PostgreSQL (production)

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

## RAG Pipeline

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

## Agent with Configuration (YAML)

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

## Write Evaluation Function

1. Create an `eval/` directory in your project if it doesn't exist.
2. Add your evaluation functions there. Each function must:
   - Be decorated with `@eval_run`
   - Return an `EvalConfig` or `SchemaEvalConfig` object

**Example:**

```python
# eval/examples.py
from rakam_systems_cli.decorators import eval_run
from rakam_systems_tools.evaluation.schema import (
    EvalConfig,
    TextInputItem,
    ClientSideMetricConfig,
    ToxicityConfig,
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
                input="Hello world", # input from ai component e.g. system_prompt or user_prompt
                output="Hello world", # output from ai component e.g. agent response
                expected_output="Hello world", # excpected  results (optional, depends on metrics requested)
                metrics=[ClientSideMetricConfig(name="relevance", score=1)],
            )
        ],
        metrics=[ToxicityConfig(name="toxicity_demo", include_reason=False)],
    )
```

---

## Run Your Evaluation

From your project root to run evaluation functions, run:

```bash
rakam eval run
```

To List runs:

```bash
rakam eval list runs

```

To View latest results:

```bash
rakam eval show
```

Compare two runs to see what changed:

```bash
# Compare by IDs
rakam eval compare --id 42 --id 45

# Save comparison to file
rakam eval compare  --id 42 --id 45 -o comparison.json

```

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

```
ModuleNotFoundError: No module named 'rakam_systems'
```

**Solution:** Make sure that rakam-system package got installed correctly:

```bash
pip install rakam-systems
```

#### 2. Missing Optional Dependencies

```
ImportError: cannot import name 'BaseAgent' from 'rakam_systems_agent'
```

**Solution:** Install the required module:

```bash
pip install rakam-systems-agent
```

#### 3. Django Not Configured

```
django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS...
```

**Solution:** Configure Django before importing Django-dependent components:

```python
import django
from django.conf import settings

settings.configure(
    INSTALLED_APPS=['rakam_systems_vectorstore.components.vectorstore'],
    DATABASES={'default': {...}}
)
django.setup()

# Now import Django-dependent components
from rakam_systems_vectorstore import ConfigurablePgVectorStore
```

#### 4. PyTorch Installation Issues

PyTorch is large (~2GB). For CPU-only installation:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 5. FAISS GPU Support

Replace CPU version with GPU version:

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

#### 6. libmagic Not Found

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

#### 7. PostgreSQL Connection Refused

Ensure PostgreSQL is running:

```bash
# Check if running
docker ps | grep postgres

# Start if not running
docker start postgres-vectorstore

# Or start with docker compose
docker compose up -d postgres
```

### Verification

#### Verify Core Installation

```python
# Test core imports
from rakam_systems_core.base import BaseComponent
from rakam_systems_core.interfaces import ToolComponent, VectorStore
print("✅ Core installed successfully!")
```

#### Verify AI Agents

```python
try:
    from rakam_systems_agent import BaseAgent
    from rakam_systems_core.interfaces.agent import AgentInput, AgentOutput
    print("✅ AI Agents installed successfully!")
except ImportError as e:
    print(f"❌ AI Agents not installed: {e}")
```

#### Verify AI Vectorstore

```python
try:
    from rakam_systems_vectorstore import (
        VectorStoreConfig,
        Node,
        VSFile,
        ConfigurableEmbeddings
    )
    print("✅ AI Vectorstore installed successfully!")
except ImportError as e:
    print(f"❌ AI Vectorstore not installed: {e}")
```

#### Verify LLM Gateway

```python
try:
    from rakam_systems_agent.components.llm_gateway import (
        OpenAIGateway,
        MistralGateway,
        LLMGatewayFactory
    )
    print("✅ LLM Gateway installed successfully!")
except ImportError as e:
    print(f"❌ LLM Gateway not installed: {e}")
```

### Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: [GitHub Issues](https://github.com/Rakam-AI/rakam_systems/issues)

---

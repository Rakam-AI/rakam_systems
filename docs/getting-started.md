---
title: Getting Started
---

# Quick Start Guide

Welcome! This guide will help you set up and use the Rakam Systems tools for local development and evaluation.

---

## Prerequisites

Before you begin, make sure you have:

- [Docker](https://docs.docker.com/get-docker/) installed
- Access credentials for the private container registry (for running evaluation)
- Python 3.8 or newer (Python 3.10 or newer in case of using agent or vectorstore module)

---

## 1. Start the Evaluation Service (required for evaluation)

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

---

## 2. Set Up the Environment

**a. Create and activate a Python virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows: venv\Scripts\activate
```

**b. Install Rakam Systems package:**

```bash
pip install rakam-systems==0.2.5rc9
```

**c. Set API keys:**

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="sk-your-api-key"
# These will come from Step 2
export EVALFRAMEWORK_URL="http://eval-service-url.com" # url of docker container
export EVALFRAMEWORK_API_KEY="your-api-token" # can be generated from '/docs' swagger-ui

# Option 2: .env file (recommended)
# Create .env with: OPENAI_API_KEY=sk-your-api-key
# ...
# Then add to your code: from dotenv import load_dotenv; load_dotenv()
```

---

## 3. Your First Agent

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

## 4. Write Your First Evaluation Function

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

## 5. Run Your Evaluation

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

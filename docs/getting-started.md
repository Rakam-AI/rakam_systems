---
title: Getting Started Guide
---

# Getting Started Guide

This guide walks you through a first end-to-end example: install Rakam Systems, create an agent, and run an evaluation. For detailed usage patterns, see the [User Guide](./user-guide/index.md).

import Prerequisites from './_partials/_prerequisites.md';

<Prerequisites />

## Set up the environment

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows: venv\Scripts\activate
```

### Install Rakam Systems

```bash
pip install rakam-systems
```

### Configure API keys

Create a `.env` file in your project root with your OpenAI API key:

```bash
# .env
OPENAI_API_KEY=sk-your-api-key
```

Then load it in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Create your first agent

Create a file named `my_first_agent.py`:

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

Run it with:

```bash
python my_first_agent.py
```

## Write an evaluation function

:::note
The evaluation service connection is configured separately. Contact us for evaluation service setup details.
:::

1. Create an `eval/` directory in your project if it doesn't exist.
2. Add your evaluation functions there. Each function must:
   - Be decorated with `@eval_run`
   - Return an `EvalConfig` object

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
                input="Hello world",
                output="Hello world",
                expected_output="Hello world",
                metrics=[ClientSideMetricConfig(name="relevance", score=1)],
            )
        ],
        metrics=[ToxicityConfig(name="toxicity_demo", include_reason=False)],
    )
```

## Run evaluations

From your project root:

```bash
rakam eval run
```

To list runs:

```bash
rakam eval list runs
```

To view latest results:

```bash
rakam eval show
```

Compare two runs to see what changed:

```bash
rakam eval compare --id 42 --id 45
```

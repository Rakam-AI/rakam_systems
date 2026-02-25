---
title: Getting Started Guide
---

# Getting Started Guide

Welcome! This guide will help you set up and use the Rakam Systems tools for local development and evaluation.

import Prerequisites from './_partials/_prerequisites.md';

<Prerequisites />

## 1. Start the Evaluation Service (required for evaluation)

The evaluation service must be running to use the evaluation features. Contact us if you need help setting it up.

## 2. Set Up the Environment

**1. Create and activate a Python virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows: venv\Scripts\activate
```

**2. Install Rakam Systems package:**

```bash
pip install rakam-systems
```

**3. Set up your API keys:**

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

:::note
The evaluation service connection is configured separately. Contact us for evaluation service setup details.
:::

## 3. Your First Agent

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

## 4. Write Your First Evaluation Function

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

## 5. Run Your Evaluation

From your project root to run evaluation functions, run:

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

# CLI Quick Start Guide

Welcome! This guide will help you set up and use the Rakam Systems CLI tools for local evaluation and monitoring.

---

## Prerequisites

Before you begin, make sure you have:

- [Docker](https://docs.docker.com/get-docker/) installed
- Access credentials for the private container registry
- Python 3.8 or newer

---

## 1. Start the Evaluation Service

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

## 2. Set Up the CLI Environment

**a. Create and activate a Python virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows: venv\Scripts\activate
```

**b. Install the CLI package:**

```bash
pip install rakam-systems-cli==0.2.5rc17
```

---

## 3. Configure the CLI

The CLI can be configured using environment variables or a `.env` file in your project root.

**Option 1: Set environment variables**

```bash
export EVALFRAMEWORK_URL="http://eval-service-url.com"
export EVALFRAMEWORK_API_KEY="your-api-token"
```

If you do not set a `base_url`, it will default to `http://localhost:8080`.

---

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
                input="Hello world",
                output="Hello world",
                expected_output="Hello world",
                metrics=[ClientSideMetricConfig(name="relevance", score=1)],
            )
        ],
        metrics=[ToxicityConfig(name="toxicity_demo", include_reason=False)],
    )
```

---

## 5. Run Your Evaluation

From your project root, run:

```bash
rakam eval run
```

---

## Key Points

- Place all evaluation scripts in the `eval/` directory
- Decorate each function with `@eval_run`
- Return a valid config object (`EvalConfig` or `SchemaEvalConfig`)
- Each input should be a `TextInputItem` or `SchemaInputItem`
- Specify the metrics to compute

---

## More Information

For advanced usage and CLI options, see the [full CLI documentation](https://github.com/Rakam-AI/evaluation-tools/blob/ft/better-process-tags/sdk/CLI.md).

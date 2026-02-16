# CLI - Step-by-Step Guide

This guide will help you set up and use the CLI tools for local evaluation and monitoring.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- Access to the private container registry (credentials required)
- Python 3.8+

---

## Start the Eval Service Stack

This will launch all required services:

```bash
docker run -d \
  --name eval-framework \
  -p 8080:8000 \
  -e OPENAI_API_KEY= YOUR_API \
  -e API_PREFIX="/eval-framework" \
  -e APP_NAME="eval-framework" \
  346k0827.c1.de1.container-registry.ovh.net/monitoring/evaluation-service:v0.2.4rc8
```

## Using the CLI package

### Setup Python Environment

First, create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# Or on Windows: venv\Scripts\activate
```

Then install the required dependencies:

```bash
pip install rakam-systems-cli==0.2.4rc15
```

### Configuration Options

The client can be configured in the following way:

**Using environment variables:** The client will automatically pick up the `EVALFRAMEWORK_URL` and `EVALFRAMEWORK_API_KEY` environment variables if they are set or it will try to read from `.env` file in the root of the project.

    ```bash
    export EVALFRAMEWORK_URL="http://eval-service-url.com"
    export EVALFRAMEWORK_API_KEY="your-api-token"
    ```

If no `base_url` is provided, it defaults to `http://localhost:8080`.

### Writing Evaluation Functions

Evaluation functions are placed in the `eval/` directory. Each function must be decorated with `@eval_run` and return an `EvalConfig` or `SchemaEvalConfig` object.

See `eval/examples.py` for complete examples. Here's a simple one:

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
    """A simple text evaluation showcasing basic client-side metric"""
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

And then run:

```bash
rakam eval run
```

**Key Points:**

- Place evaluation scripts in the `eval/` directory
- Use `@eval_run` to decorate your function
- Return a valid config object (`EvalConfig` or `SchemaEvalConfig`)
- Each input should be a `TextInputItem` or `SchemaInputItem`
- Specify metrics to be computed

---

## CLI Usage

Refer to https://github.com/Rakam-AI/evaluation-tools/blob/ft/better-process-tags/sdk/CLI.md for more information

---
title: Evaluation
---

# Evaluation

The evaluation service must be running to use evaluation features. Contact us if you need help setting it up.

Configure access to the evaluation service in your `.env`:

```bash
# Evaluation service access
EVALFRAMEWORK_URL="http://eval-service-url.com"   # URL of the evaluation service
EVALFRAMEWORK_API_KEY="your-api-token"             # Generate from the /docs Swagger UI
```

## Write an evaluation function

Create an `eval/` directory in your project and add evaluation functions decorated with `@eval_run`. Each function returns an `EvalConfig` or `SchemaEvalConfig`.

### Text evaluation

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

### Schema evaluation

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

### Client-side metrics

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

### Probabilistic evaluation

Use `maybe_*` methods to run evaluations on a sample of requests, reducing load on the evaluation service:

```python
from rakam_systems_tools.evaluation import DeepEvalClient

client = DeepEvalClient()

# Runs approximately 10% of the time
client.maybe_text_eval(data=data, metrics=metrics, chance=0.1)
```

### Error handling

By default, the evaluation client returns a dictionary with an `"error"` key on failure. Set `raise_exception=True` to raise instead:

```python
from rakam_systems_tools.evaluation import DeepEvalClient

client = DeepEvalClient()

try:
    result = client.text_eval(data=data, metrics=metrics, raise_exception=True)
except requests.RequestException as e:
    print(f"An error occurred: {e}")
```

## Run evaluations

Install the CLI package:

```bash
pip install rakam-systems-cli
```

### Execute evaluations

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

### View results

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

### Compare runs

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

### Tag runs

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

### List runs and evaluations

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

# Rakam Eval CLI

A CLI for running LLM evaluations and tracking quality over time.

---

## Quick Start

A typical workflow is:

1. Write eval function

```bash
edit eval/my_eval.py 
```

2. Run evaluation

```bash
rakam eval run
```

3. View results

```bash
rakam eval show
```

---

## Installation

```bash
pip install rakam-systems-cli
```

---

## Writing Evaluations

Create an `eval/` directory in your project. Each evaluation function must:

- Be decorated with `@eval_run`
- Return an `EvalConfig` object

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

## User Guide

### Listing evaluations

```bash
rakam eval list evals
```

This shows all functions decorated with `@eval_run` in the `eval/` directory.

### Listing runs

This shows all runs hosted on the evaluation server.

```bash
rakam eval list runs
```

### Comparing runs

Compare two runs to see what changed:

```bash
# Compare by IDs
rakam eval compare --id 42 --id 45

# Save comparison to file
rakam eval compare --id 42 --id 45 -o comparison.json
```

---

## Command Reference

<details>
<summary>Full command reference (click to expand)</summary>

### `rakam eval list evals`

```
Usage: rakam eval list evals [OPTIONS] [DIRECTORY]

 List evaluations (functions decorated with @eval_run).

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│   directory      [DIRECTORY]  Directory to scan (default: ./eval)            │
│                               [default: eval]                                │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --recursive  -r        Recursively search for Python files                   │
│ --help                 Show this message and exit.                           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `rakam eval list runs`

```
Usage: rakam eval list runs [OPTIONS]

 List runs (newest first).

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --limit   -l      INTEGER  Max number of runs [default: 20]                  │
│ --offset          INTEGER  Pagination offset [default: 0]                    │
│ --help                     Show this message and exit.                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `rakam eval run`

```
Usage: rakam eval run [OPTIONS] [DIRECTORY]

 Execute evaluations (functions decorated with @eval_run).

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│   directory      [DIRECTORY]  Directory to scan (default: ./eval)            │
│                               [default: eval]                                │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --recursive   -r            Recursively search for Python files              │
│ --dry-run                   Only list functions without executing them       │
│ --save-runs                 Save each run result to a JSON file              │
│ --output-dir          PATH  Directory where run results are saved            │
│                             [default: eval_runs]                             │
│ --help                      Show this message and exit.                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `rakam eval show`

```
Usage: rakam eval show [OPTIONS]

 Show a run by ID or tag. Without arguments, shows the most recent run.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --id    -i      INTEGER  Run ID                                              │
│ --tag   -t      TEXT     Run tag                                             │
│ --raw                    Print raw JSON instead of formatted output          │
│ --help                   Show this message and exit.                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `rakam eval compare`

```
Usage: rakam eval compare [OPTIONS]

 Compare two evaluation runs.

 Default: unified git diff

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --tag           -t      TEXT     Run tag                                     │
│ --id            -i      INTEGER  Run ID                                      │
│ --summary                        Show summary diff only                      │
│ --side-by-side                   Show side-by-side diff (git)                │
│ --help                           Show this message and exit.                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `rakam eval tag`

```
Usage: rakam eval tag [OPTIONS]

 Assign a tag to a run or delete a tag.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --id      -i      INTEGER  Run ID                                            │
│ --tag     -t      TEXT     Tag to assign to the run                          │
│ --delete          TEXT     Delete a tag                                      │
│ --help                     Show this message and exit.                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `rakam eval metrics list`

```
Usage: rakam eval metrics list [OPTIONS] [DIRECTORY]

 List all metric types used by loaded eval configs.

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│   directory      [DIRECTORY]  Directory to scan (default: ./eval)            │
│                               [default: eval]                                │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --recursive  -r        Recursively search for Python files                   │
│ --help                 Show this message and exit.                           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

</details>

## License

See main project LICENSE file.

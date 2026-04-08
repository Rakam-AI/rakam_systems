---
title: Observability
---

# Observability

Rakam Systems provides two complementary observability layers:

| Layer                 | Package               | What it does                                                                                     |
| --------------------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| **TrackingManager**   | `rakam-systems-core`  | Lightweight I/O capture for local debugging and CSV/JSON export                                  |
| **EvaluationTracker** | `rakam-systems-tools` | Backend-agnostic tracing to Langfuse or MLflow — sessions, users, token counts, scores, datasets |

---

## TrackingManager (core)

Built-in input/output tracking for debugging and evaluation:

```python
from rakam_systems_core.tracking import TrackingManager, track_method, TrackingMixin

class MyAgent(TrackingMixin, BaseAgent):
    @track_method()
    async def arun(self, input_data, deps=None):
        return await super().arun(input_data, deps)

# Enable tracking
agent.enable_tracking(output_dir="./tracking")

# Export tracking data
agent.export_tracking_data(format='csv')
agent.export_tracking_data(format='json')

# Get statistics
stats = agent.get_tracking_statistics()
```

---

## EvaluationTracker (tools)

`EvaluationTracker` is an abstract interface in `rakam_systems_tools.evaluation.observability` that provides a single API for both **Langfuse** and **MLflow**. Switch backends by changing one argument to `create_tracker()`.

### Installation

```bash
# Langfuse only
pip install 'rakam-systems-tools[langfuse]'

# MLflow only
pip install 'rakam-systems-tools[mlflow]'

# Both
pip install 'rakam-systems-tools[observability]'
```

### Quick start

```python
from rakam_systems_tools.evaluation.observability import create_tracker

# Langfuse backend
tracker = create_tracker(
    "langfuse",
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="http://localhost:3000",       # or set LANGFUSE_HOST
)

# MLflow backend — use experiment_name to create/resolve the experiment automatically
tracker = create_tracker(
    "mlflow",
    tracking_uri="http://localhost:5000",  # or set MLFLOW_TRACKING_URI
    experiment_name="my-experiment",       # creates experiment if it doesn't exist
)

# Log a single-span trace
trace_id = tracker.log_trace(
    name="qa_pipeline",
    input={"question": "What is the capital of France?"},
    output={"answer": "Paris"},
)

# Flush pending exports before the process exits
tracker.flush()
```

---

### Backend comparison

| Feature                        | Langfuse                                            | MLflow                                                       |
| ------------------------------ | --------------------------------------------------- | ------------------------------------------------------------ |
| `log_trace()`                  | Native generation span                              | Root `CHAIN` span + child `LLM` span                         |
| `start_trace()` nested tracing | `start_as_current_observation()`                    | `start_span()` (root = new trace)                            |
| `flush()`                      | `client.flush()`                                    | OTel `TracerProvider.force_flush()`                          |
| Sessions                       | Native (`session_id` field)                         | Emulated via `mlflow.trace.session` metadata                 |
| Users                          | Native (`user_id` field)                            | Emulated via `mlflow.trace.user` metadata                    |
| Token / cost                   | `usage_details` + `cost_details` on generation span | `mlflow.chat.tokenUsage` + `mlflow.llm.cost` span attributes |
| `log_score()`                  | `create_score()`                                    | `log_feedback()` with `AssessmentSource`                     |
| `get_session()`                | `api.sessions.get()`                                | Searches traces by metadata                                  |
| `list_sessions()`              | `api.sessions.list()`                               | Scans distinct `mlflow.trace.session` values                 |
| `create_dataset()`             | Supported                                           | **Not supported** (raises `NotImplementedError`)             |
| `add_dataset_item()`           | Supported                                           | **Not supported**                                            |
| `evaluate_traces()`            | **Not supported** (raises `NotImplementedError`)    | `mlflow.genai.evaluate()`                                    |
| Tags                           | String list on trace                                | String keys on trace                                         |
| Metadata                       | Key/value on trace                                  | Key/value in trace metadata dict                             |

---

### `create_tracker()`

```python
from rakam_systems_tools.evaluation.observability import create_tracker

tracker = create_tracker(backend, **kwargs)
```

| Parameter  | Type                       | Description                                |
| ---------- | -------------------------- | ------------------------------------------ |
| `backend`  | `"langfuse"` \| `"mlflow"` | Which backend to instantiate               |
| `**kwargs` |                            | Passed directly to the backend constructor |

**Langfuse kwargs**

| Kwarg        | Env var fallback      | Description                 |
| ------------ | --------------------- | --------------------------- |
| `public_key` | `LANGFUSE_PUBLIC_KEY` | Langfuse project public key |
| `secret_key` | `LANGFUSE_SECRET_KEY` | Langfuse project secret key |
| `host`       | `LANGFUSE_HOST`       | Langfuse server URL         |

**MLflow kwargs**

| Kwarg             | Env var fallback       | Description                                                                              |
| ----------------- | ---------------------- | ---------------------------------------------------------------------------------------- |
| `tracking_uri`    | `MLFLOW_TRACKING_URI`  | MLflow tracking server URL                                                               |
| `experiment_name` | —                      | Experiment name — created automatically if it doesn't exist; resolves to `experiment_id` |
| `experiment_id`   | `MLFLOW_EXPERIMENT_ID` | MLflow experiment ID (use `experiment_name` instead when the ID is not yet known)        |

`tracker.experiment_id` exposes the resolved experiment ID after construction (useful for building UI links).

---

### `flush()`

Flush any pending trace exports to the backend before the process exits. Call this at the end of scripts that log traces.

```python
tracker.flush()
```

- **Langfuse** — calls `client.flush()` to drain the background HTTP queue.
- **MLflow** — calls `TracerProvider.force_flush()` (OTel-based in MLflow 3.x).

`flush()` is a no-op on backends where exports are already synchronous.

---

### Nested tracing with `start_trace()`

Use `start_trace()` when you need a **parent/child span hierarchy** — for example, a RAG pipeline with separate retrieval and generation spans.

```python
with tracker.start_trace(
    name="rag_pipeline",
    input={"question": question},
    session_id="session-abc",
    user_id="user-alice",
    tags=["rag", "production"],
) as trace:

    # Child span — document retrieval
    with trace.span("retrieve", input={"query": question}, span_type="retriever") as span:
        docs = retriever.get(question)
        span.set_output({"docs": docs, "count": len(docs)})

    # Child span — LLM generation with token/cost tracking
    with trace.span("generate", input={"question": question, "context": docs}, span_type="generation") as span:
        answer, usage = llm.call(question, docs)
        span.set_output({"answer": answer}, usage=usage)

    # Set the root trace output and attach an automated score
    trace.set_output({"answer": answer})
    trace.add_score("quality", 0.95, source_type="CODE")

# Access the trace ID after the context exits
print(trace.trace_id)
```

**`start_trace()` parameters**

| Parameter    | Type                  | Description                            |
| ------------ | --------------------- | -------------------------------------- |
| `name`       | `str`                 | Root span / trace name shown in the UI |
| `input`      | `dict`                | Input payload for the root span        |
| `session_id` | `str` \| `None`       | Groups this trace into a session       |
| `user_id`    | `str` \| `None`       | Links trace to a specific user         |
| `tags`       | `list[str]` \| `None` | Searchable string labels               |

Returns a context manager that yields a **`TraceHandle`**.

---

#### `TraceHandle`

Represents the active root trace inside a `start_trace()` block.

| Member                                                           | Description                                                                  |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `trace.trace_id`                                                 | The backend trace/run ID (accessible both inside and after the `with` block) |
| `trace.set_output(output)`                                       | Set the root span output dict                                                |
| `trace.add_score(name, value, comment=None, source_type="CODE")` | Attach a score to the trace                                                  |
| `trace.span(name, input, span_type="span")`                      | Context manager — creates a child span, yields a `SpanHandle`                |

---

#### `SpanHandle`

Represents an active child span inside a `trace.span()` block.

| Member                                | Description                                                        |
| ------------------------------------- | ------------------------------------------------------------------ |
| `span.set_output(output, usage=None)` | Set span output; pass `usage` dict to record token counts and cost |

The `usage` dict accepts the same keys as `log_trace()`: `model`, `input_tokens`, `output_tokens`, `total_tokens`, `input_cost`, `output_cost`, `total_cost`.

---

#### `span_type` values

The `span_type` argument controls how the span is rendered in each backend's UI.

| `span_type`        | Langfuse `as_type` | MLflow `SpanType` | Use for                               |
| ------------------ | ------------------ | ----------------- | ------------------------------------- |
| `"span"` (default) | `span`             | `UNKNOWN`         | Generic step                          |
| `"chain"`          | `chain`            | `CHAIN`           | Multi-step pipeline                   |
| `"retriever"`      | `retriever`        | `RETRIEVER`       | Document retrieval                    |
| `"generation"`     | `generation`       | `LLM`             | LLM call (enables token/cost display) |
| `"tool"`           | `tool`             | `TOOL`            | Tool / function call                  |
| `"agent"`          | `agent`            | `AGENT`           | Agent reasoning step                  |
| `"embedding"`      | `embedding`        | `EMBEDDING`       | Embedding call                        |

---

### `log_trace()`

Log a single LLM call or pipeline run as a named trace. Use this for flat (non-nested) tracing; use `start_trace()` for hierarchical spans.

```python
trace_id = tracker.log_trace(
    name="qa_pipeline",
    input={"question": "What is the capital of France?", "system": "Answer concisely."},
    output={"answer": "Paris"},
    metadata={"category": "geography", "difficulty": "easy"},
    tags=["direct", "geography", "easy"],
    session_id="session-abc",
    user_id="user-alice",
    usage={
        "model": "gpt-4o-mini",
        "input_tokens": 45,
        "output_tokens": 3,
        "total_tokens": 48,
        "input_cost": 0.0000068,
        "output_cost": 0.0000018,
        "total_cost": 0.0000086,
    },
)
```

**Parameters**

| Parameter    | Type                  | Description                             |
| ------------ | --------------------- | --------------------------------------- |
| `name`       | `str`                 | Trace name shown in the UI              |
| `input`      | `dict`                | Input payload (question, prompt, etc.)  |
| `output`     | `dict`                | Output payload (answer, response, etc.) |
| `metadata`   | `dict` \| `None`      | Arbitrary key-value metadata            |
| `tags`       | `list[str]` \| `None` | Searchable string labels                |
| `session_id` | `str` \| `None`       | Groups traces into a session            |
| `user_id`    | `str` \| `None`       | Links trace to a specific user          |
| `usage`      | `dict` \| `None`      | Token counts and cost — see below       |

**`usage` dict keys**

| Key             | Type    | Description                                           |
| --------------- | ------- | ----------------------------------------------------- |
| `model`         | `str`   | Model name (e.g. `"gpt-4o-mini"`)                     |
| `input_tokens`  | `int`   | Prompt / input token count                            |
| `output_tokens` | `int`   | Completion / output token count                       |
| `total_tokens`  | `int`   | Total tokens (computed automatically if omitted)      |
| `input_cost`    | `float` | Cost of input tokens in USD                           |
| `output_cost`   | `float` | Cost of output tokens in USD                          |
| `total_cost`    | `float` | Total cost in USD (computed automatically if omitted) |

When `usage` is provided, Langfuse logs the span as a `generation` (enabling the token/cost display in the UI) and MLflow sets the `mlflow.chat.tokenUsage` and `mlflow.llm.cost` span attributes on the `LLM` span.

---

### `fetch_traces()`

Search traces with optional filters.

```python
# By name
traces = tracker.fetch_traces(name="qa_pipeline", limit=20)

# By tag
traces = tracker.fetch_traces(tags=["geography"], limit=10)

# By session
traces = tracker.fetch_traces(session_id="session-abc")

# By user
traces = tracker.fetch_traces(user_id="user-alice", limit=50)

# Combined
traces = tracker.fetch_traces(session_id="session-abc", user_id="user-alice")
```

**Parameters**

| Parameter    | Type                  | Default | Description           |
| ------------ | --------------------- | ------- | --------------------- |
| `name`       | `str` \| `None`       | `None`  | Filter by trace name  |
| `tags`       | `list[str]` \| `None` | `None`  | Filter by tags        |
| `session_id` | `str` \| `None`       | `None`  | Filter to one session |
| `user_id`    | `str` \| `None`       | `None`  | Filter to one user    |
| `limit`      | `int`                 | `50`    | Maximum results       |

Returns a `list[dict]` — each dict is a trace record.

---

### `get_trace()`

Fetch a single trace by ID.

```python
trace = tracker.get_trace(trace_id)
# trace is a dict — fields vary by backend
```

---

### `log_score()`

Attach a numeric score to a trace (human feedback, automated metrics, or LLM-judge).

```python
tracker.log_score(
    trace_id=trace_id,
    name="human_quality",
    value=0.9,
    comment="Correct and concise.",
    source_type="HUMAN",       # "HUMAN" | "LLM_JUDGE" | "CODE"
)

tracker.log_score(
    trace_id=trace_id,
    name="conciseness",
    value=0.75,
    source_type="CODE",
)
```

---

### Sessions

Sessions group related traces (e.g. all turns in a conversation, or all questions for a prompting variant). Both backends support session filtering via `fetch_traces(session_id=...)`.

```python
# Log traces into a session
for question in questions:
    tracker.log_trace(..., session_id="variant-direct-run-1234")

# Fetch all traces in a session
session = tracker.get_session("variant-direct-run-1234")
traces = session["traces"]   # list of trace dicts

# Browse recent sessions
sessions = tracker.list_sessions(limit=20)
# [{"session_id": "...", "trace_count": 12}, ...]
```

**Langfuse** stores session_id natively on the trace. Sessions are visible in the Langfuse UI under the **Sessions** tab.

**MLflow** emulates sessions via the `mlflow.trace.session` metadata key. `get_session()` and `list_sessions()` query traces by that metadata field.

---

### User tracking

Assign traces to individual users so per-user activity is visible in the backend UI.

```python
tracker.log_trace(
    name="qa_pipeline",
    input={...},
    output={...},
    user_id="user-alice",
)

# Filter traces by user
traces = tracker.fetch_traces(user_id="user-alice", limit=100)
```

**Langfuse** stores `user_id` natively. Users appear under the **Users** tab.

**MLflow** emulates user tracking via the `mlflow.trace.user` metadata key.

---

### Datasets (Langfuse only)

Datasets store reusable input/expected-output pairs for offline evaluation.

```python
# Create a dataset
ds_name = tracker.create_dataset(
    name="qa-eval-dataset",
    description="12 Q&A pairs for variant comparison",
    metadata={"version": "1"},
)

# Add items
item_id = tracker.add_dataset_item(
    dataset_name="qa-eval-dataset",
    input={"question": "What is the capital of France?"},
    expected_output={"answer": "Paris"},
    metadata={"category": "geography", "difficulty": "easy"},
    item_id="geo1",   # optional stable ID
)

# Inspect
dataset = tracker.get_dataset("qa-eval-dataset")

# List all datasets
datasets = tracker.list_datasets()
```

> MLflow raises `NotImplementedError` for all dataset methods. Use MLflow artifact logging or experiment parameters instead.

---

### Evaluate traces (MLflow only)

Run LLM-judge scorers over a set of traces using `mlflow.genai.evaluate()`.

```python
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety

results = tracker.evaluate_traces(
    trace_ids=["run-abc", "run-def", ...],
    scorers=[
        Correctness(model="openai:/gpt-4o-mini"),
        RelevanceToQuery(model="openai:/gpt-4o-mini"),
        Safety(model="openai:/gpt-4o-mini"),
    ],
)
# results is a list of dicts with scorer output per trace
```

> Langfuse raises `NotImplementedError` for `evaluate_traces()`. Use the Langfuse UI evaluators or `log_score()` for programmatic scoring.

---

### Token and cost tracking

Extract usage from your LLM response and pass it to `log_trace()` or `span.set_output()`:

```python
# OpenAI
response = client.chat.completions.create(model="gpt-4o-mini", ...)
u = response.usage
trace_id = tracker.log_trace(
    name="chat",
    input={"messages": messages},
    output={"answer": response.choices[0].message.content},
    usage={
        "model": "gpt-4o-mini",
        "input_tokens": u.prompt_tokens,
        "output_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
    },
)

# Anthropic
response = client.messages.create(model="claude-haiku-4-5-20251001", ...)
u = response.usage
trace_id = tracker.log_trace(
    name="chat",
    input={"messages": messages},
    output={"answer": response.content[0].text},
    usage={
        "model": "claude-haiku-4-5-20251001",
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
    },
)
```

The same `usage` dict is accepted by `span.set_output()` inside a `start_trace()` block.

**How each backend stores it**

_Langfuse_ — switches `as_type` to `"generation"` and passes:

```
usage_details = {"input": input_tokens, "output": output_tokens, "total": total_tokens}
cost_details  = {"input": input_cost,   "output": output_cost,   "total": total_cost}
```

_MLflow_ — sets span attributes on the `LLM` span using `set_attribute()`:

```
mlflow.chat.tokenUsage → {"input_tokens": N, "output_tokens": N, "total_tokens": N}
mlflow.llm.cost        → {"input_cost": N,   "output_cost": N,   "total_cost": N}
mlflow.llm.model       → "gpt-4o-mini"
```

---

### Demo scripts

Four runnable demo scripts are included in `tools/src/rakam_systems_tools/scripts/`:

| Script                    | Backend  | What it demonstrates                                                                                       |
| ------------------------- | -------- | ---------------------------------------------------------------------------------------------------------- |
| `langfuse_demo.py`        | Langfuse | `log_trace`, `log_score`, sessions, users, datasets, `fetch_traces`, `get_trace`                           |
| `langfuse_nested_demo.py` | Langfuse | Nested RAG pipeline: `start_trace` → `trace.span` (retriever + generation), `add_score`, `flush`           |
| `mlflow_demo.py`          | MLflow   | `start_trace`, `trace.span`, `log_score`, `evaluate_traces`, sessions, users — all via tracker abstraction |
| `mlflow_nested_demo.py`   | MLflow   | Nested RAG pipeline: `start_trace` → `trace.span` (retriever + generation), `add_score`, `flush`           |

All scripts use a 12-question Q&A dataset across three prompting variants (`direct`, `chain_of_thought`, `expert`) and support OpenAI, Anthropic, or mock LLM providers.

**Run the Langfuse demos**

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="http://localhost:3000"
export OPENAI_API_KEY="sk-..."   # optional; falls back to Anthropic, then mock

cd tools
uv run src/rakam_systems_tools/scripts/langfuse_demo.py
uv run src/rakam_systems_tools/scripts/langfuse_nested_demo.py
```

**Run the MLflow demos**

```bash
mlflow server --host 0.0.0.0 --port 5000

export MLFLOW_TRACKING_URI="http://localhost:5000"
export OPENAI_API_KEY="sk-..."   # optional

cd tools
uv run src/rakam_systems_tools/scripts/mlflow_demo.py
uv run src/rakam_systems_tools/scripts/mlflow_nested_demo.py
```

**LLM provider selection (all demos)**

| Priority | Condition                       | Provider used                           |
| -------- | ------------------------------- | --------------------------------------- |
| 1        | `OPENAI_API_KEY` is set         | OpenAI (`gpt-4o-mini` by default)       |
| 2        | `ANTHROPIC_API_KEY` is set      | Anthropic (`claude-haiku-4-5-20251001`) |
| 3        | `MOCK_LLM=1` or neither key set | Built-in mock (canned answers)          |

---

## Related

- [User Guide — Evaluation](../user-guide/evaluation.md)
- [CLI reference](../cli/index.md)
- [Environment variables](./environment.md)

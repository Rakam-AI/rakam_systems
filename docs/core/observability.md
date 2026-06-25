---
title: Observability
---

# Observability

Rakam Systems provides two complementary observability layers:

| Layer | Package | What it does |
|---|---|---|
| **TrackingManager** | `rakam-systems-core` | Lightweight I/O capture for local debugging and CSV/JSON export |
| **EvaluationTracker** | `rakam-systems-tools` | Backend-agnostic tracing to Langfuse or MLflow — sessions, users, token counts, scores, datasets |

---

## TrackingManager (core)

The `TrackingManager` in `rakam_systems_core.tracking` records method inputs and outputs for local debugging. Enable it on any agent or component that mixes in `TrackingMixin`:

```python
from rakam_systems_core.tracking import TrackingManager, track_method, TrackingMixin

class MyAgent(TrackingMixin, BaseAgent):
    @track_method()
    async def arun(self, input_data, deps=None):
        return await super().arun(input_data, deps)

agent.enable_tracking(output_dir="./tracking")
agent.export_tracking_data(format='csv')
stats = agent.get_tracking_statistics()
```

Data is written to local files only — no external service required.

---

## EvaluationTracker (tools)

For production tracing, install `rakam-systems-tools` with the `observability` extra and use `EvaluationTracker` via the `create_tracker()` factory:

```python
from rakam_systems_tools.evaluation.observability import create_tracker

tracker = create_tracker("langfuse", public_key="pk-lf-...", secret_key="sk-lf-...")
# or
tracker = create_tracker("mlflow", tracking_uri="http://localhost:5000")

trace_id = tracker.log_trace(
    name="qa_pipeline",
    input={"question": "What is the capital of France?"},
    output={"answer": "Paris"},
    session_id="session-abc",
    user_id="user-alice",
)
tracker.log_score(trace_id, "quality", 0.9, source_type="HUMAN")
```

Supports Langfuse and MLflow as interchangeable backends. Switch by changing the `backend` argument — the rest of the code stays the same.

See the [Developer Guide — Observability](../developer-guide/tracking.md) for the full API reference, backend comparison, session tracking, user tracking, token/cost tracking, datasets, and demo scripts.

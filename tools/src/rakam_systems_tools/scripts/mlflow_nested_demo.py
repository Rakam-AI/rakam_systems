"""MLflow nested tracing demo — RAG pipeline.

Uses the ``EvaluationTracker`` abstraction from
``rakam_systems_tools.evaluation.observability`` with the MLflow backend.

Trace structure produced:
  rag_pipeline  [CHAIN]
  ├── retrieve  [RETRIEVER]
  └── generate  [LLM, token usage + cost attributes]

Setup:
    mlflow server --host 0.0.0.0 --port 5000
    export MLFLOW_TRACKING_URI="http://localhost:5000"

Run:
    cd tools
    uv run src/rakam_systems_tools/scripts/mlflow_nested_demo.py
"""

import os
import random

from rakam_systems_tools.evaluation.observability import create_tracker

# ---------------------------------------------------------------------------
# Fake knowledge base + mock LLM
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "qa-variants-demo"

DOCS = {
    "capital": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
    "language": ["French is spoken in France.", "German is spoken in Germany."],
    "default": ["No specific document found."],
}

QUESTIONS = [
    "What is the capital of France?",
    "What language is spoken in Germany?",
    "Who painted the Mona Lisa?",
]

SESSION_ID = "mlflow-nested-demo-session"
USER_ID = os.environ.get("MLFLOW_USER_ID", "demo-user")


def retrieve(query: str) -> list[str]:
    for key, docs in DOCS.items():
        if key in query.lower():
            return docs
    return DOCS["default"]


def generate(question: str, docs: list[str]) -> tuple[str, dict]:
    prompt = f"Context: {' '.join(docs)}\nQuestion: {question}"
    if "capital" in question.lower():
        answer = docs[0].split(" is ")[0] if docs else "Unknown"
    elif "language" in question.lower():
        answer = docs[0].split(" is ")[1].strip(" .") if docs else "Unknown"
    else:
        answer = "I don't know."

    input_tokens = len(prompt.split())
    output_tokens = len(answer.split())
    usage = {
        "model": "mock-llm",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": round(input_tokens * 0.00000015, 8),
        "output_cost": round(output_tokens * 0.0000006, 8),
    }
    return answer, usage


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run(tracker, question: str) -> None:
    print(f"\n  Q: {question}")

    with tracker.start_trace(
        name="rag_pipeline",
        input={"question": question},
        session_id=SESSION_ID,
        user_id=USER_ID,
        tags=["rag", "demo"],
    ) as trace:

        with trace.span("retrieve", input={"query": question}, span_type="retriever") as span:
            docs = retrieve(question)
            span.set_output({"docs": docs, "count": len(docs)})

        with trace.span("generate", input={"question": question, "context": docs}, span_type="generation") as span:
            answer, usage = generate(question, docs)
            span.set_output({"answer": answer}, usage=usage)

        trace.set_output({"answer": answer})
        trace.add_score("mock_quality", round(random.uniform(0.6, 1.0), 2))

    print(f"  A: {answer}")
    print(f"  trace_id={trace.trace_id}  tokens={usage['total_tokens']}")


def main() -> None:
    tracker = create_tracker(
        "mlflow",
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        experiment_name=EXPERIMENT_NAME,
    )

    print("=== MLflow nested tracing demo ===")
    print(f"experiment : {EXPERIMENT_NAME}  (id={tracker.experiment_id})")
    print(f"session_id : {SESSION_ID}\n")

    for question in QUESTIONS:
        run(tracker, question)

    tracker.flush()
    print("\nDone — check the MLflow UI to see nested spans.")


if __name__ == "__main__":
    main()

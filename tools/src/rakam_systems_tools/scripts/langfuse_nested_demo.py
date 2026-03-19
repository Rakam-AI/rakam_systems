"""Langfuse nested tracing demo — RAG pipeline.

Uses the ``EvaluationTracker`` abstraction from
``rakam_systems_tools.evaluation.observability`` with the Langfuse backend.

Trace structure produced:
  rag_pipeline  [chain]
  ├── retrieve  [retriever]
  └── generate  [generation, token usage + cost]

Setup:
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_HOST="http://localhost:3000"

Run:
    cd tools
    uv run src/rakam_systems_tools/scripts/langfuse_nested_demo.py
"""

import random

from rakam_systems_tools.evaluation.observability import create_tracker
import os

LANGFUSE_PUBLIC_KEY = os.getenv(
    "LANGFUSE_PUBLIC_KEY", "pk-lf-ae842f9c-a24d-4957-9d6c-50c3a64af81a")
LANGFUSE_SECRET_KEY = os.getenv(
    "LANGFUSE_SECRET_KEY", "sk-lf-1ab503e9-f196-4c01-81f6-7f9c8d490705")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST") or os.getenv(
    "LANGFUSE_BASE_URL", "http://localhost:3000")

# ---------------------------------------------------------------------------
# Fake knowledge base + mock LLM
# ---------------------------------------------------------------------------
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

def run(tracker, question: str, session_id: str, user_id: str) -> None:
    print(f"\n  Q: {question}")

    with tracker.start_trace(
        name="rag_pipeline",
        input={"question": question},
        session_id=session_id,
        user_id=user_id,
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
    tracker = create_tracker("langfuse", public_key=LANGFUSE_PUBLIC_KEY,
                             secret_key=LANGFUSE_SECRET_KEY,
                             host=LANGFUSE_HOST,)

    session_id = "langfuse-nested-demo-session"
    user_id = "demo-user"

    print("=== Langfuse nested tracing demo ===")
    print(f"session_id: {session_id}\n")

    for question in QUESTIONS:
        run(tracker, question, session_id=session_id, user_id=user_id)

    tracker.flush()
    print("\nDone — check the Langfuse UI to see nested spans.")


if __name__ == "__main__":
    main()

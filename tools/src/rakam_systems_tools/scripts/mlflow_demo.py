"""
MLflow Tracing & Evaluation Demo  (via EvaluationTracker)
==========================================================
Same Q&A variant demo, now routing all tracing through the unified
``rakam_systems_tools.evaluation.observability`` abstraction.

What goes through the tracker
------------------------------
  tracker.start_trace()      → root CHAIN span per pipeline call
  trace.span()               → child LLM generation span
  trace.set_output()         → set root span output
  trace.trace_id             → captured trace ID (replaces mlflow.get_last_active_trace_id)
  tracker.log_score()        → human feedback
  tracker.evaluate_traces()  → LLM-judge evaluation
  tracker.fetch_traces()     → simple name-based search
  tracker.get_session()      → session inspection
  tracker.list_sessions()    → browse sessions
  tracker.flush()            → wait for async export

What stays as direct MLflow calls (no tracker equivalent)
----------------------------------------------------------
  mlflow.anthropic.autolog()     zero-code auto-tracing for real Anthropic client
  mlflow.update_current_trace()  key/value tag enrichment for rich filter queries
  mlflow.entities.Expectation    ground-truth annotation (not in tracker API)
  MlflowClient.log_expectation   ground-truth annotation
  MlflowClient.set_trace_tag     fallback tag setting
  MlflowClient.search_traces     rich filter-string search

Dataset
-------
12 questions across 6 categories, 3 difficulty levels.

Sessions
--------
Each prompting variant runs in its own session via tracker.start_trace(session_id=...).

LLM Providers (priority order)
--------------------------------
  1. OpenAI   — set OPENAI_API_KEY  (model: gpt-4o-mini by default)
  2. Anthropic — set ANTHROPIC_API_KEY  (model: claude-haiku-4-5, with autolog)
  3. Mock     — set MOCK_LLM=1 or leave both keys unset

Setup
-----
  mlflow server --host 0.0.0.0 --port 5000
  uv run mlflow_demo.py
  open http://localhost:5000
"""

import os
import time
from typing import Optional

import mlflow
import mlflow.anthropic
from mlflow import MlflowClient

from rakam_systems_tools.evaluation.observability import create_tracker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "qa-variants-demo"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Dataset – 12 questions, 6 categories, 3 difficulty levels
# ---------------------------------------------------------------------------

QA_DATASET = [
    # ── Geography ────────────────────────────────────────────────────────
    {"id": "geo1",  "category": "geography",   "difficulty": "easy",   "question": "What is the capital of France?",                                  "expected": "Paris"},
    {"id": "geo2",  "category": "geography",   "difficulty": "medium", "question": "What is the longest river in the world?",                         "expected": "The Nile River (approximately 6,650 km / 4,130 miles)"},
    # ── Science ──────────────────────────────────────────────────────────
    {"id": "sci1",  "category": "science",     "difficulty": "easy",   "question": "What is the boiling point of water in Celsius at sea level?",     "expected": "100 degrees Celsius"},
    {"id": "sci2",  "category": "science",     "difficulty": "easy",   "question": "What is known as the powerhouse of the cell?",                    "expected": "The mitochondria"},
    {"id": "sci3",  "category": "science",     "difficulty": "medium", "question": "What is the approximate speed of light in metres per second?",    "expected": "299,792,458 metres per second (approximately 3×10⁸ m/s)"},
    {"id": "sci4",  "category": "science",     "difficulty": "hard",   "question": "What is the half-life of Carbon-14 used in radiocarbon dating?",  "expected": "Approximately 5,730 years"},
    # ── History ──────────────────────────────────────────────────────────
    {"id": "hist1", "category": "history",     "difficulty": "easy",   "question": "In what year did World War II end?",                              "expected": "1945"},
    {"id": "hist2", "category": "history",     "difficulty": "medium", "question": "Who was the first person to walk on the Moon?",                   "expected": "Neil Armstrong (Apollo 11, July 20, 1969)"},
    # ── Literature ───────────────────────────────────────────────────────
    {"id": "lit1",  "category": "literature",  "difficulty": "easy",   "question": "Who wrote the play Hamlet?",                                      "expected": "William Shakespeare"},
    {"id": "lit2",  "category": "literature",  "difficulty": "medium", "question": "In George Orwell's 1984, what is the name of the ruling party?",  "expected": "The Party (INGSOC / English Socialism)"},
    # ── Math ─────────────────────────────────────────────────────────────
    {"id": "math1", "category": "math",        "difficulty": "easy",   "question": "What is the square root of 144?",                                 "expected": "12"},
    # ── Astronomy ────────────────────────────────────────────────────────
    {"id": "ast1",  "category": "astronomy",   "difficulty": "medium", "question": "How far is the Moon from Earth on average in kilometres?",         "expected": "Approximately 384,400 kilometres (238,855 miles)"},
]

# ---------------------------------------------------------------------------
# Prompting variants
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "direct": {
        "system": "Answer the question in one concise sentence.",
        "description": "Minimal, single-sentence answers",
    },
    "chain_of_thought": {
        "system": (
            "Think step by step before answering. "
            "Show your reasoning, then state the final answer clearly."
        ),
        "description": "Explicit step-by-step reasoning before the answer",
    },
    "expert": {
        "system": (
            "You are a world-class expert. Provide an accurate, authoritative answer "
            "with one or two sentences of relevant context or background."
        ),
        "description": "Authoritative answer with brief supporting context",
    },
}

# ---------------------------------------------------------------------------
# Mock answers
# ---------------------------------------------------------------------------

MOCK_ANSWERS: dict[str, dict[str, str]] = {
    "direct": {
        "geo1": "Paris.", "geo2": "The Nile River, at approximately 6,650 km.",
        "sci1": "100°C.", "sci2": "The mitochondria.",
        "sci3": "Approximately 299,792,458 metres per second.", "sci4": "Approximately 5,730 years.",
        "hist1": "1945.", "hist2": "Neil Armstrong on July 20, 1969.",
        "lit1": "William Shakespeare.", "lit2": "The Party (INGSOC).",
        "math1": "12.", "ast1": "Approximately 384,400 kilometres.",
    },
    "chain_of_thought": {
        "geo1": "France is a country in Western Europe. Its political and cultural hub has historically been in the north. Therefore, the capital of France is Paris.",
        "geo2": "River length is measured from source to mouth. The Nile (~6,650 km) and Amazon (~6,400 km) are the two main candidates. Most measurements place the Nile slightly longer.",
        "sci1": "The boiling point depends on atmospheric pressure. At 1 atm the liquid-to-gas phase transition of water occurs at exactly 100 degrees Celsius.",
        "sci2": "Cells require energy in the form of ATP. ATP is produced through cellular respiration. The organelle responsible is the mitochondria, hence 'powerhouse of the cell'.",
        "sci3": "Light travels through a vacuum at a fixed universal constant c, defined as exactly 299,792,458 metres per second (approximately 3×10⁸ m/s).",
        "sci4": "Radioactive decay follows an exponential law. Carbon-14 decays to Nitrogen-14 by beta emission. Half of any sample decays in approximately 5,730 years.",
        "hist1": "The war in Europe ended May 8, 1945 (V-E Day). The war in the Pacific ended September 2, 1945 (V-J Day). Therefore, WWII fully ended in 1945.",
        "hist2": "Apollo 11 was the first crewed lunar landing. Commander Neil Armstrong stepped onto the Moon on July 20, 1969, making him the first person to walk on the Moon.",
        "lit1": "Hamlet is one of the most celebrated tragedies in English literature. It was written around 1600–1601 by William Shakespeare.",
        "lit2": "In Orwell's 1984, Oceania is governed by a totalitarian party called 'The Party' whose ideology is INGSOC (English Socialism).",
        "math1": "We need n such that n×n=144. Testing: 10²=100, 11²=121, 12²=144. The square root of 144 is 12.",
        "ast1": "The Moon orbits Earth in an elliptical path. The average Earth-Moon distance is approximately 384,400 kilometres (238,855 miles).",
    },
    "expert": {
        "geo1": "Paris is the capital and most populous city of France, located on the Seine River. It has been the country's political and cultural centre since the early medieval period.",
        "geo2": "The Nile River in northeastern Africa is generally considered the world's longest at ~6,650 km, flowing through 11 countries. Some studies argue the Amazon may be longer depending on measurement method.",
        "sci1": "At standard atmospheric pressure (101.325 kPa), water undergoes its phase transition at exactly 100°C (212°F / 373.15 K). This value decreases at higher altitudes.",
        "sci2": "Mitochondria are membrane-bound organelles in eukaryotic cells that generate most ATP through oxidative phosphorylation, earning the nickname 'powerhouse of the cell'.",
        "sci3": "The speed of light in vacuum (c) is exactly 299,792,458 m/s (~3×10⁸ m/s). It is the universal speed limit and underpins Einstein's special relativity.",
        "sci4": "Carbon-14 has a half-life of ~5,730 years (±40 yr). This predictable decay rate is the basis of radiocarbon dating, reliable for organic material up to ~50,000 years old.",
        "hist1": "WWII ended in 1945: Germany surrendered May 8 (V-E Day), Japan surrendered September 2 (V-J Day) after the atomic bombings. An estimated 70–85 million casualties.",
        "hist2": "Neil Armstrong, commander of Apollo 11, became the first human on the Moon at 02:56 UTC on July 20, 1969, saying 'That's one small step for [a] man, one giant leap for mankind.'",
        "lit1": "Hamlet was written by William Shakespeare circa 1599–1601 and explores themes of revenge, mortality, and moral corruption through its Danish prince protagonist.",
        "lit2": "In Orwell's Nineteen Eighty-Four (1949), 'The Party' rules Oceania via INGSOC ideology, enforcing control through the Thought Police, Newspeak, and constant surveillance.",
        "math1": "The square root of 144 is 12, since 12²=144. Both +12 and −12 are technically square roots; the principal (positive) square root is 12.",
        "ast1": "The mean Earth-Moon distance is ~384,400 km (one lunar distance). The orbit is elliptical, ranging from ~356,500 km at perigee to ~406,700 km at apogee.",
    },
}

# ---------------------------------------------------------------------------
# Mock LLM client  (no direct MLflow calls)
# ---------------------------------------------------------------------------


class MockAnthropicClient:
    """Mimics anthropic.Anthropic() with canned responses. Span creation is handled
    by the pipeline via tracker.start_trace() / trace.span()."""

    class _Content:
        def __init__(self, text: str):
            self.text = text

    class _Response:
        def __init__(self, text: str, model: str):
            self.content = [MockAnthropicClient._Content(text)]
            self.model = model
            n_words = len(text.split())
            self.usage = {"input_tokens": n_words // 2, "output_tokens": n_words}

    class _Messages:
        def __init__(self, answers: dict):
            self._answers = answers

        def create(self, model, max_tokens, system, messages, **_):
            question = messages[0]["content"] if messages else ""
            if "step by step" in system:
                variant_key = "chain_of_thought"
            elif "world-class" in system:
                variant_key = "expert"
            else:
                variant_key = "direct"

            qa_map = self._answers.get(variant_key, {})
            answer = f"[mock] {question[:40]}"
            for item in QA_DATASET:
                if item["question"] == question:
                    answer = qa_map.get(item["id"], answer)
                    break

            difficulty_delay = {"easy": 0.04, "medium": 0.07, "hard": 0.12}
            item_match = next((i for i in QA_DATASET if i["question"] == question), {})
            time.sleep(difficulty_delay.get(item_match.get("difficulty", "easy"), 0.05))
            return MockAnthropicClient._Response(answer, model)

    def __init__(self, mock_answers: dict):
        self._answers = mock_answers

    @property
    def messages(self):
        return self._Messages(self._answers)


# ---------------------------------------------------------------------------
# Q&A pipeline  (uses tracker.start_trace + trace.span — no @mlflow.trace)
# ---------------------------------------------------------------------------


def build_qa_fn(tracker, client, provider: str, model_name: str):
    def qa_pipeline(
        question: str,
        system_prompt: str,
        variant: str,
        category: str,
        difficulty: str,
        question_id: str,
        session_id: str = "",
        user_id: str = "",
    ) -> tuple[str, str]:
        with tracker.start_trace(
            name="qa_pipeline",
            input={"question": question, "system_prompt": system_prompt},
            session_id=session_id or None,
            user_id=user_id or None,
            tags=[variant, category, difficulty],
        ) as trace:

            with trace.span(
                "llm_call",
                input={"question": question, "system_prompt": system_prompt},
                span_type="generation",
            ) as llm_span:
                if provider == "openai":
                    response = client.chat.completions.create(
                        model=model_name,
                        max_tokens=256,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                    )
                    answer = response.choices[0].message.content or ""
                    u = response.usage
                    inp_tok = getattr(u, "prompt_tokens", 0) or 0
                    out_tok = getattr(u, "completion_tokens", 0) or 0
                else:  # anthropic or mock
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=256,
                        system=system_prompt,
                        messages=[{"role": "user", "content": question}],
                    )
                    answer = response.content[0].text
                    u = response.usage
                    inp_tok = (u.get("input_tokens", 0) if isinstance(u, dict) else getattr(u, "input_tokens", 0)) or 0
                    out_tok = (u.get("output_tokens", 0) if isinstance(u, dict) else getattr(u, "output_tokens", 0)) or 0

                llm_span.set_output(
                    {"answer": answer},
                    usage={
                        "model": model_name,
                        "input_tokens": inp_tok,
                        "output_tokens": out_tok,
                        "total_tokens": inp_tok + out_tok,
                    },
                )

            # MLflow-specific key/value tags for rich filter-string queries in the UI.
            # update_current_trace() is called while the root span context is active.
            mlflow.update_current_trace(
                tags={
                    "variant": variant,
                    "category": category,
                    "difficulty": difficulty,
                    "question_id": question_id,
                    "question": question[:120],
                    "prompt_length": str(len(system_prompt)),
                    "provider": provider,
                    "model": model_name,
                    "response_length": str(len(answer)),
                    "word_count": str(len(answer.split())),
                    "input_tokens": str(inp_tok),
                    "output_tokens": str(out_tok),
                    "total_tokens": str(inp_tok + out_tok),
                },
            )

            trace.set_output({"answer": answer})
            tid = trace.trace_id

        return answer, tid

    return qa_pipeline


# ---------------------------------------------------------------------------
# Phase 1 – Generate all traces
# ---------------------------------------------------------------------------


def run_all_variants(
    qa_fn,
    mlf_client: MlflowClient,
    session_prefix: str = "demo",
    user_id: Optional[str] = None,
) -> dict[str, list[str]]:
    trace_ids: dict[str, list[str]] = {v: [] for v in VARIANTS}
    session_ids = {v: f"{session_prefix}-{v}" for v in VARIANTS}
    effective_user_id = user_id or session_prefix
    total = len(VARIANTS) * len(QA_DATASET)
    done = 0

    print(f"\n── Phase 1: Generating traces ({total} total) ──────────────────────")
    print(f"  Sessions: {', '.join(session_ids.values())}")
    print(f"  User ID : {effective_user_id}")

    for item in QA_DATASET:
        qid = item["id"]
        question = item["question"]
        expected = item["expected"]
        category = item["category"]
        diff = item["difficulty"]

        print(f"\n  [{category}/{diff}] {question}")

        for variant_name, vcfg in VARIANTS.items():
            answer, tid = qa_fn(
                question=question,
                system_prompt=vcfg["system"],
                variant=variant_name,
                category=category,
                difficulty=diff,
                question_id=qid,
                session_id=session_ids[variant_name],
                user_id=effective_user_id,
            )

            if not tid:
                print(f"    [{variant_name}] WARNING: no trace ID captured")
                continue

            trace_ids[variant_name].append(tid)

            # log_expectation is not in the tracker API — use MlflowClient directly.
            try:
                mlf_client.log_expectation(
                    trace_id=tid,
                    expectation=mlflow.entities.Expectation(
                        value={"expected_answer": expected}
                    ),
                )
            except Exception:
                mlf_client.set_trace_tag(tid, "expected_answer", expected)

            done += 1
            short = answer.replace("\n", " ")[:85]
            print(f"    [{variant_name:>18}] {short}…  ({done}/{total})")

    return trace_ids


# ---------------------------------------------------------------------------
# Phase 2 – Evaluate  (via tracker.evaluate_traces)
# ---------------------------------------------------------------------------


def evaluate_traces(
    trace_ids: dict[str, list[str]],
    experiment_id: str,
    tracker,
) -> None:
    all_ids = [tid for ids in trace_ids.values() for tid in ids]
    print(f"\n── Phase 2: Evaluating {len(all_ids)} traces via tracker ──────────────────")

    expected_by_qid = {item["id"]: item["expected"] for item in QA_DATASET}

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        judge_model = f"openai:/{OPENAI_MODEL}"
    elif anthropic_key:
        judge_model = f"anthropic:/{ANTHROPIC_MODEL}"
    else:
        judge_model = None

    if judge_model is None:
        print("  Skipping LLM scorers — set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable.")
        print(f"\n  Or evaluate via MCP:")
        print(f"    experiment_id = {experiment_id!r}")
        print(f"    trace_ids     = {','.join(all_ids[:3])}…")
        print(f"    scorers       = 'correctness,relevance_to_query,safety'")
        return

    try:
        from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety

        mlflow.set_experiment(EXPERIMENT_NAME)

        scorers = [
            Correctness(model=judge_model),
            RelevanceToQuery(model=judge_model),
            Safety(model=judge_model),
        ]
        print(f"  Scorers: Correctness, RelevanceToQuery, Safety  (judge={judge_model})")

        print("  Fetching trace objects via tracker…")
        data = []
        for tid in all_ids:
            trace = tracker.get_trace(tid)
            tags = {}
            if hasattr(trace, "info"):
                tags = trace.info.tags or {}
            elif isinstance(trace, dict):
                tags = trace.get("info", {}).get("tags", {})
            qid = tags.get("question_id", "")
            data.append({
                "trace": trace,
                "expectations": {"expected_response": expected_by_qid.get(qid, "")},
            })

        results = tracker.evaluate_traces(data, scorers=scorers)

        if results and isinstance(results[0], dict):
            print(f"\n  Evaluation complete — {len(results)} result rows")
        else:
            print(f"\n  Evaluation complete.")

    except Exception as exc:
        print(f"  tracker.evaluate_traces() error: {exc}")
        print(f"\n  Run via MCP instead:")
        print(f"    experiment_id = {experiment_id!r}")
        print(f"    trace_ids     = {','.join(all_ids[:3])}…")


# ---------------------------------------------------------------------------
# Phase 3 – Search examples
# ---------------------------------------------------------------------------


def search_examples(
    trace_ids: dict[str, list[str]],
    experiment_id: str,
    tracker,
    mlf_client: MlflowClient,
) -> None:
    print("\n── Phase 3: Search examples ────────────────────────────────────────")

    print("\n  [tracker.fetch_traces] name='qa_pipeline', limit=5:")
    try:
        results = tracker.fetch_traces(name="qa_pipeline", limit=5)
        print(f"    → {len(results)} traces returned")
    except Exception as exc:
        print(f"    → {exc}")

    # Rich tag-based searches need MlflowClient — tracker.fetch_traces() only supports
    # name/session/user/tag-label filtering, not MLflow's arbitrary filter-string syntax.
    rich_searches = [
        ("hard questions",           "tags.difficulty = 'hard'"),
        ("science category",         "tags.category = 'science'"),
        ("chain_of_thought variant", "tags.variant = 'chain_of_thought'"),
        ("fast responses (≤40ms)",   "execution_time_ms <= 40"),
    ]
    print("\n  [mlf_client.search_traces] rich filter-string queries:")
    for label, fstr in rich_searches:
        try:
            results = mlf_client.search_traces(
                locations=[experiment_id],
                filter_string=fstr,
                max_results=5,
            )
            print(f"\n  {label!r}  →  {len(results)} traces")
            for t in results[:3]:
                tags = t.info.tags or {}
                q = tags.get("question", "?")[:55]
                wc = tags.get("word_count", "?")
                dur = getattr(t.info, "execution_duration", "?")
                print(f"    {t.info.trace_id}  wc={wc:<4} dur={dur}ms  '{q}'")
        except Exception as exc:
            print(f"  {label!r}: {exc}")


# ---------------------------------------------------------------------------
# Phase 4 – Human feedback via tracker.log_score
# ---------------------------------------------------------------------------


def add_sample_feedback(trace_ids: dict[str, list[str]], tracker) -> None:
    print("\n── Phase 4: Adding sample human feedback via tracker.log_score ─────")

    quality_scores = {
        "direct":           (1.0, "Correct and maximally concise."),
        "chain_of_thought": (0.6, "Correct but unnecessarily long for a simple factual question."),
        "expert":           (0.9, "Correct with helpful geographic context."),
    }

    for variant_name, ids in trace_ids.items():
        if not ids:
            continue
        tid = ids[0]
        score, rationale = quality_scores[variant_name]
        try:
            tracker.log_score(
                trace_id=tid,
                name="human_quality",
                value=score,
                comment=rationale,
                source_type="HUMAN",
            )
            print(f"  {variant_name:<22} human_quality={score}")
        except Exception as exc:
            print(f"  {variant_name} feedback error: {exc}")


# ---------------------------------------------------------------------------
# Phase 4b – Session inspection via tracker
# ---------------------------------------------------------------------------


def show_sessions(session_prefix: str, tracker) -> None:
    print("\n── Phase 4b: Session inspection via tracker ────────────────────────")

    print("\n  [tracker.list_sessions] limit=10:")
    try:
        sessions = tracker.list_sessions(limit=10)
        print(f"    → {len(sessions)} sessions found")
        for s in sessions[:5]:
            sid = s.get("session_id", "?") if isinstance(s, dict) else "?"
            cnt = s.get("trace_count", "?") if isinstance(s, dict) else "?"
            print(f"      {sid}  ({cnt} traces)")
    except Exception as exc:
        print(f"    → {exc}")

    for variant_name in list(VARIANTS.keys())[:2]:
        sid = f"{session_prefix}-{variant_name}"
        print(f"\n  [tracker.get_session] session_id={sid!r}:")
        try:
            session = tracker.get_session(sid)
            traces = session.get("traces", []) if isinstance(session, dict) else []
            print(f"    → {len(traces)} traces in session")
        except Exception as exc:
            print(f"    → {exc}")


# ---------------------------------------------------------------------------
# Phase 5 – Custom scorer (MCP instructions only)
# ---------------------------------------------------------------------------


def show_custom_scorer_instructions(experiment_id: str) -> None:
    print("\n── Phase 5: Custom LLM judge (register via MCP) ────────────────────")
    print(
        f"\n  mcp__mlflow-mcp__register_llm_judge(\n"
        f"      name='conciseness_judge',\n"
        f"      experiment_id={experiment_id!r},\n"
        f"      instructions=(\n"
        f"          'Evaluate whether {{{{ outputs }}}} answers the question in '\n"
        f"          '2 sentences or fewer. Return yes or no with a brief reason.'\n"
        f"      ),\n"
        f"  )\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    force_mock = os.getenv("MOCK_LLM") == "1"

    if force_mock or (not openai_key and not anthropic_key):
        provider, use_mock, model_name = "mock", True, ANTHROPIC_MODEL
    elif openai_key:
        provider, use_mock, model_name = "openai", False, OPENAI_MODEL
    else:
        provider, use_mock, model_name = "anthropic", False, ANTHROPIC_MODEL

    session_prefix = f"mlflow-{int(time.time())}"
    user_id: Optional[str] = os.getenv("MLFLOW_USER_ID") or session_prefix

    # Set up experiment — needed for experiment_id (used by tracker + MlflowClient)
    # and for mlflow.anthropic.autolog() to know the destination.
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    mlf_client = MlflowClient(tracking_uri=TRACKING_URI)

    tracker = create_tracker(
        "mlflow",
        tracking_uri=TRACKING_URI,
        experiment_id=experiment_id,
    )

    categories = sorted({i["category"] for i in QA_DATASET})
    difficulties = {d: sum(1 for i in QA_DATASET if i["difficulty"] == d) for d in ["easy", "medium", "hard"]}

    print("MLflow Tracing & Evaluation Demo  (via EvaluationTracker)")
    print(f"  Tracking URI  : {TRACKING_URI}")
    print(f"  Experiment    : {EXPERIMENT_NAME}  (id={experiment_id})")
    print(f"  UI / Traces   : {TRACKING_URI}/#/experiments/{experiment_id}/traces")
    print(f"  Variants      : {', '.join(VARIANTS)}")
    print(f"  Questions     : {len(QA_DATASET)}  ({', '.join(categories)})")
    print(f"  Difficulty    : {', '.join(f'{k}={v}' for k, v in difficulties.items())}")
    print(f"  Total traces  : {len(VARIANTS) * len(QA_DATASET)}")
    print(f"  LLM provider  : {provider}  (model: {model_name})")
    print(f"  Session prefix: {session_prefix}")
    print(f"  User ID       : {user_id}")

    if use_mock:
        client = MockAnthropicClient(MOCK_ANSWERS)
    elif provider == "openai":
        import openai as openai_sdk
        client = openai_sdk.OpenAI(api_key=openai_key)
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        mlflow.anthropic.autolog()  # no tracker equivalent — kept as direct call

    qa_fn = build_qa_fn(tracker, client, provider=provider, model_name=model_name)

    trace_ids = run_all_variants(qa_fn, mlf_client, session_prefix=session_prefix, user_id=user_id)
    tracker.flush()
    evaluate_traces(trace_ids, experiment_id, tracker)
    search_examples(trace_ids, experiment_id, tracker, mlf_client)
    add_sample_feedback(trace_ids, tracker)
    show_sessions(session_prefix, tracker)
    show_custom_scorer_instructions(experiment_id)

    all_ids = [tid for ids in trace_ids.values() for tid in ids]
    print("\n── Done ────────────────────────────────────────────────────────────")
    print(f"  Logged {len(all_ids)} traces across {len(VARIANTS)} variants × {len(QA_DATASET)} questions")
    for variant, ids in trace_ids.items():
        sid = f"{session_prefix}-{variant}"
        print(f"    {variant:<22} {len(ids)} traces  session={sid}")
    print(f"\n  Explore: {TRACKING_URI}/#/experiments/{experiment_id}/traces")
    print()


if __name__ == "__main__":
    main()

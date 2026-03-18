"""
Langfuse Tracing & Evaluation Demo  (via EvaluationTracker)
============================================================
Parallel to the MLflow demo: runs three Q&A prompting *variants* across a
diverse dataset and logs everything through the unified EvaluationTracker
abstraction — backend is "langfuse".

Because Langfuse does not have a decorator-based auto-tracing API, every
LLM call is logged explicitly via tracker.log_trace().

EvaluationTracker features demonstrated
----------------------------------------
  tracker.log_trace()        log each Q&A call as a named trace
  tracker.log_score()        human feedback + automated code-side scores
  tracker.create_dataset()   create a reusable evaluation dataset
  tracker.add_dataset_item() populate the dataset with Q&A pairs
  tracker.get_dataset()      verify dataset contents
  tracker.fetch_traces()     simple name / tag search
  tracker.get_trace()        fetch individual trace by ID
  tracker.evaluate_traces()  raises NotImplementedError (Langfuse limitation)

Dataset
-------
12 questions across 6 categories (geography, science, history, literature,
math, astronomy) and 3 difficulty levels (easy / medium / hard).

Modes
-----
  Real   export ANTHROPIC_API_KEY=sk-ant-...  then  uv run langfuse_demo.py
  Mock   unset key or set MOCK_LLM=1          (uses canned answers)

Setup
-----
  export LANGFUSE_PUBLIC_KEY="pk-lf-..."
  export LANGFUSE_SECRET_KEY="sk-lf-..."
  export LANGFUSE_HOST="http://langfuse-web:3000"
  uv run langfuse_demo.py
"""

import os
import time

from rakam_systems_tools.evaluation.tracking import create_tracker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-ae842f9c-a24d-4957-9d6c-50c3a64af81a")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-1ab503e9-f196-4c01-81f6-7f9c8d490705")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")
MODEL = "claude-haiku-4-5-20251001"
DATASET_NAME = "qa-variants-dataset"
TRACE_NAME = "qa_pipeline"

# ---------------------------------------------------------------------------
# Dataset – 12 questions, 6 categories, 3 difficulty levels
# ---------------------------------------------------------------------------

QA_DATASET = [
    # ── Geography ────────────────────────────────────────────────────────
    {
        "id": "geo1",
        "category": "geography",
        "difficulty": "easy",
        "question": "What is the capital of France?",
        "expected": "Paris",
    },
    {
        "id": "geo2",
        "category": "geography",
        "difficulty": "medium",
        "question": "What is the longest river in the world?",
        "expected": "The Nile River (approximately 6,650 km / 4,130 miles)",
    },
    # ── Science ──────────────────────────────────────────────────────────
    {
        "id": "sci1",
        "category": "science",
        "difficulty": "easy",
        "question": "What is the boiling point of water in Celsius at sea level?",
        "expected": "100 degrees Celsius",
    },
    {
        "id": "sci2",
        "category": "science",
        "difficulty": "easy",
        "question": "What is known as the powerhouse of the cell?",
        "expected": "The mitochondria",
    },
    {
        "id": "sci3",
        "category": "science",
        "difficulty": "medium",
        "question": "What is the approximate speed of light in metres per second?",
        "expected": "299,792,458 metres per second (approximately 3×10⁸ m/s)",
    },
    {
        "id": "sci4",
        "category": "science",
        "difficulty": "hard",
        "question": "What is the half-life of Carbon-14 used in radiocarbon dating?",
        "expected": "Approximately 5,730 years",
    },
    # ── History ──────────────────────────────────────────────────────────
    {
        "id": "hist1",
        "category": "history",
        "difficulty": "easy",
        "question": "In what year did World War II end?",
        "expected": "1945",
    },
    {
        "id": "hist2",
        "category": "history",
        "difficulty": "medium",
        "question": "Who was the first person to walk on the Moon?",
        "expected": "Neil Armstrong (Apollo 11, July 20, 1969)",
    },
    # ── Literature ───────────────────────────────────────────────────────
    {
        "id": "lit1",
        "category": "literature",
        "difficulty": "easy",
        "question": "Who wrote the play Hamlet?",
        "expected": "William Shakespeare",
    },
    {
        "id": "lit2",
        "category": "literature",
        "difficulty": "medium",
        "question": "In George Orwell's 1984, what is the name of the ruling party?",
        "expected": "The Party (INGSOC / English Socialism)",
    },
    # ── Math ─────────────────────────────────────────────────────────────
    {
        "id": "math1",
        "category": "math",
        "difficulty": "easy",
        "question": "What is the square root of 144?",
        "expected": "12",
    },
    # ── Astronomy ────────────────────────────────────────────────────────
    {
        "id": "ast1",
        "category": "astronomy",
        "difficulty": "medium",
        "question": "How far is the Moon from Earth on average in kilometres?",
        "expected": "Approximately 384,400 kilometres (238,855 miles)",
    },
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
# Mock answers – canned responses that intentionally vary by variant
# ---------------------------------------------------------------------------

MOCK_ANSWERS: dict[str, dict[str, str]] = {
    "direct": {
        "geo1":  "Paris.",
        "geo2":  "The Nile River, at approximately 6,650 km.",
        "sci1":  "100°C.",
        "sci2":  "The mitochondria.",
        "sci3":  "Approximately 299,792,458 metres per second.",
        "sci4":  "Approximately 5,730 years.",
        "hist1": "1945.",
        "hist2": "Neil Armstrong on July 20, 1969.",
        "lit1":  "William Shakespeare.",
        "lit2":  "The Party (INGSOC).",
        "math1": "12.",
        "ast1":  "Approximately 384,400 kilometres.",
    },
    "chain_of_thought": {
        "geo1": (
            "France is a country in Western Europe with a long-standing centralised "
            "government. Its political, cultural, and administrative hub has historically "
            "been in the north of the country. Therefore, the capital of France is Paris."
        ),
        "geo2": (
            "River length is measured from source to mouth. The two main candidates are "
            "the Nile in Africa (~6,650 km) and the Amazon in South America (~6,400 km). "
            "Most traditional measurements place the Nile slightly longer, so the answer "
            "is the Nile River."
        ),
        "sci1": (
            "The boiling point depends on atmospheric pressure. At standard sea-level "
            "pressure (1 atm / 101.325 kPa) the liquid-to-gas phase transition of water "
            "occurs at exactly 100 degrees Celsius."
        ),
        "sci2": (
            "Cells require energy in the form of ATP to carry out their functions. "
            "ATP is produced through a process called cellular respiration. The organelle "
            "responsible for generating most of a cell's ATP is the mitochondria, hence "
            "it is called the powerhouse of the cell."
        ),
        "sci3": (
            "Light travels through a vacuum at a fixed, universal constant. "
            "This constant, denoted c, is defined as exactly 299,792,458 metres per second "
            "(approximately 3×10⁸ m/s)."
        ),
        "sci4": (
            "Radioactive decay follows an exponential law. Carbon-14 is a radioactive "
            "isotope that decays to Nitrogen-14 by beta emission. Scientists have measured "
            "that half of any given sample of C-14 decays in approximately 5,730 years, "
            "which is its half-life."
        ),
        "hist1": (
            "World War II involved fighting across multiple theatres. The war in Europe "
            "ended with Germany's unconditional surrender on May 8, 1945 (V-E Day). "
            "The war in the Pacific ended with Japan's surrender on September 2, 1945 "
            "(V-J Day). Therefore, WWII fully ended in 1945."
        ),
        "hist2": (
            "NASA's Apollo program aimed to land humans on the Moon. Apollo 11 was the "
            "first crewed mission to achieve a lunar landing. Commander Neil Armstrong "
            "stepped onto the Moon's surface on July 20, 1969, making him the first "
            "person to walk on the Moon."
        ),
        "lit1": (
            "Hamlet is one of the most celebrated tragedies in the English language. "
            "It was written during the Elizabethan era, a golden age of English theatre. "
            "The play was composed around 1600–1601 by William Shakespeare."
        ),
        "lit2": (
            "George Orwell wrote the dystopian novel 1984 (published 1949). In the "
            "book, Oceania is governed by a totalitarian government. The ruling party "
            "calls itself 'The Party' and its ideology is called INGSOC (English "
            "Socialism)."
        ),
        "math1": (
            "We need a number n such that n × n = 144. Testing: 10² = 100, 11² = 121, "
            "12² = 144. Since 12 × 12 = 144, the square root of 144 is 12."
        ),
        "ast1": (
            "The Moon orbits Earth in an elliptical path, so the distance varies. "
            "The average (mean) Earth-Moon distance, known as a lunar distance, is "
            "approximately 384,400 kilometres (238,855 miles)."
        ),
    },
    "expert": {
        "geo1": (
            "Paris is the capital and most populous city of France, located on the "
            "Seine River in northern France. It has been the country's political and "
            "cultural centre since the early medieval period."
        ),
        "geo2": (
            "The Nile River in northeastern Africa is generally considered the world's "
            "longest river at approximately 6,650 km (4,130 mi), flowing northward "
            "through 11 countries before emptying into the Mediterranean Sea. Note: "
            "some studies argue the Amazon may be longer depending on measurement method."
        ),
        "sci1": (
            "At standard atmospheric pressure (101.325 kPa / 1 atm), water undergoes "
            "its liquid-to-vapour phase transition at exactly 100 °C (212 °F / 373.15 K). "
            "This value decreases at higher altitudes where ambient pressure is lower."
        ),
        "sci2": (
            "Mitochondria are membrane-bound organelles found in the cytoplasm of "
            "eukaryotic cells. They generate most of the cell's ATP through oxidative "
            "phosphorylation, earning the nickname 'powerhouse of the cell'."
        ),
        "sci3": (
            "The speed of light in a vacuum (c) is a fundamental physical constant "
            "defined as exactly 299,792,458 metres per second (~3×10⁸ m/s). It is the "
            "universal speed limit and underpins Einstein's theory of special relativity."
        ),
        "sci4": (
            "Carbon-14 (¹⁴C) is a naturally occurring radioactive isotope with a "
            "half-life of approximately 5,730 years (±40 years). This predictable decay "
            "rate is the basis of radiocarbon dating, which can reliably date organic "
            "material up to ~50,000 years old."
        ),
        "hist1": (
            "World War II ended in 1945: Germany surrendered unconditionally on May 8 "
            "(V-E Day), and Japan surrendered on September 2 (V-J Day) following the "
            "atomic bombings of Hiroshima and Nagasaki. It remains the deadliest conflict "
            "in human history, with an estimated 70–85 million casualties."
        ),
        "hist2": (
            "Neil Armstrong, commander of NASA's Apollo 11 mission, became the first "
            "human to walk on the Moon at 02:56 UTC on July 20, 1969. His first words "
            "upon stepping onto the lunar surface were: 'That's one small step for "
            "[a] man, one giant leap for mankind.'"
        ),
        "lit1": (
            "Hamlet, Prince of Denmark was written by William Shakespeare circa "
            "1599–1601 and is considered one of the greatest works in world literature. "
            "It explores themes of revenge, mortality, and moral corruption through its "
            "Danish prince protagonist."
        ),
        "lit2": (
            "In Orwell's Nineteen Eighty-Four (1949), Oceania is ruled by 'The Party' "
            "whose ideology is called INGSOC (English Socialism). The Party is led by "
            "the figurehead Big Brother and enforces total control through the Thought "
            "Police, Newspeak, and constant surveillance."
        ),
        "math1": (
            "The square root of 144 is 12, since 12² = 144. Both +12 and −12 are "
            "technically square roots; the principal (positive) square root is 12."
        ),
        "ast1": (
            "The mean Earth-Moon distance is approximately 384,400 km (238,855 mi), "
            "defined as one lunar distance (LD). Because the Moon's orbit is elliptical, "
            "this ranges from ~356,500 km at perigee to ~406,700 km at apogee."
        ),
    },
}

# ---------------------------------------------------------------------------
# Mock LLM call
# ---------------------------------------------------------------------------


def mock_llm_call(question: str, system: str, variant_key: str) -> str:
    """Return a canned answer with a simulated latency."""
    qa_map = MOCK_ANSWERS.get(variant_key, {})
    answer = f"[mock] {question[:40]}"
    for item in QA_DATASET:
        if item["question"] == question:
            answer = qa_map.get(item["id"], answer)
            difficulty_delay = {"easy": 0.04, "medium": 0.07, "hard": 0.12}
            time.sleep(difficulty_delay.get(item.get("difficulty", "easy"), 0.05))
            break
    return answer


def real_llm_call(client, question: str, system: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Phase 1 – Generate and log all traces via tracker.log_trace
# ---------------------------------------------------------------------------

def run_all_variants(tracker, use_mock: bool, client=None) -> dict[str, list[str]]:
    trace_ids: dict[str, list[str]] = {v: [] for v in VARIANTS}
    total = len(VARIANTS) * len(QA_DATASET)
    done = 0

    print(f"\n── Phase 1: Generating & logging traces ({total} total) ─────────────")

    for item in QA_DATASET:
        qid = item["id"]
        question = item["question"]
        category = item["category"]
        diff = item["difficulty"]

        print(f"\n  [{category}/{diff}] {question}")

        for variant_name, vcfg in VARIANTS.items():
            system = vcfg["system"]
            t0 = time.time()

            if use_mock:
                answer = mock_llm_call(question, system, variant_name)
            else:
                answer = real_llm_call(client, question, system)

            elapsed_ms = int((time.time() - t0) * 1000)

            # log_trace is the primary tracing call for Langfuse
            tid = tracker.log_trace(
                name=TRACE_NAME,
                input={"question": question, "system": system},
                output={"answer": answer},
                metadata={
                    "model": MODEL,
                    "variant": variant_name,
                    "category": category,
                    "difficulty": diff,
                    "question_id": qid,
                    "prompt_length": len(system),
                    "response_length": len(answer),
                    "word_count": len(answer.split()),
                    "elapsed_ms": elapsed_ms,
                },
                tags=[variant_name, category, diff],
            )

            trace_ids[variant_name].append(tid)
            done += 1
            short = answer.replace("\n", " ")[:85]
            print(f"    [{variant_name:>18}] {short}…  tid={tid[:8]}  ({done}/{total})")

    return trace_ids


# ---------------------------------------------------------------------------
# Phase 2 – Create evaluation dataset via tracker
# ---------------------------------------------------------------------------

def create_eval_dataset(tracker) -> None:
    print(f"\n── Phase 2: Creating evaluation dataset '{DATASET_NAME}' ─────────────")

    try:
        ds_name = tracker.create_dataset(
            name=DATASET_NAME,
            description="Q&A pairs for variant comparison evaluation",
            metadata={"source": "qa-variants-demo", "version": "1"},
        )
        print(f"  Dataset created: {ds_name!r}")
    except Exception as exc:
        print(f"  create_dataset: {exc}  (may already exist)")

    added = 0
    for item in QA_DATASET:
        try:
            item_id = tracker.add_dataset_item(
                dataset_name=DATASET_NAME,
                input={"question": item["question"]},
                expected_output={"answer": item["expected"]},
                metadata={
                    "id": item["id"],
                    "category": item["category"],
                    "difficulty": item["difficulty"],
                },
                item_id=item["id"],
            )
            added += 1
        except Exception as exc:
            print(f"  add_dataset_item {item['id']}: {exc}")

    print(f"  Added {added}/{len(QA_DATASET)} items")

    try:
        ds = tracker.get_dataset(DATASET_NAME)
        items = ds.get("items", []) if isinstance(ds, dict) else getattr(ds, "items", [])
        print(f"  Verified — dataset has {len(items)} items")
    except Exception as exc:
        print(f"  get_dataset: {exc}")


# ---------------------------------------------------------------------------
# Phase 3 – Human + code-side feedback via tracker.log_score
# ---------------------------------------------------------------------------

def add_feedback(trace_ids: dict[str, list[str]], tracker) -> None:
    print("\n── Phase 3: Logging scores via tracker.log_score ───────────────────")

    # Human quality scores for the first question (geo1 — capital of France)
    human_scores = {
        "direct":           (1.0, "Correct and maximally concise."),
        "chain_of_thought": (0.6, "Correct but unnecessarily verbose for a simple question."),
        "expert":           (0.9, "Correct with helpful geographic context."),
    }

    for variant_name, ids in trace_ids.items():
        if not ids:
            continue
        tid = ids[0]
        score, rationale = human_scores[variant_name]

        try:
            tracker.log_score(
                trace_id=tid,
                name="human_quality",
                value=score,
                comment=rationale,
                source_type="HUMAN",
            )
            print(f"  {variant_name:<22} human_quality={score:.1f}  ({rationale[:50]})")
        except Exception as exc:
            print(f"  {variant_name} human score error: {exc}")

    # Code-side conciseness score for every trace (word_count-based)
    print("\n  Code-side conciseness scores (word_count <= 10 → 1.0, else scaled):")
    all_ids = [tid for ids in trace_ids.values() for tid in ids]
    scored = 0
    for variant_name, ids in trace_ids.items():
        for i, tid in enumerate(ids[:3]):  # score first 3 per variant as sample
            item = QA_DATASET[i]
            answer = MOCK_ANSWERS[variant_name].get(item["id"], "")
            wc = len(answer.split())
            conciseness = min(1.0, 10 / wc) if wc > 0 else 0.0
            try:
                tracker.log_score(
                    trace_id=tid,
                    name="conciseness",
                    value=round(conciseness, 3),
                    comment=f"word_count={wc}",
                    source_type="CODE",
                )
                scored += 1
            except Exception as exc:
                print(f"    {tid[:8]} conciseness error: {exc}")
    print(f"  Logged conciseness scores for {scored} sample traces")


# ---------------------------------------------------------------------------
# Phase 4 – Fetch traces via tracker.fetch_traces / tracker.get_trace
# ---------------------------------------------------------------------------

def search_examples(trace_ids: dict[str, list[str]], tracker) -> None:
    print("\n── Phase 4: Fetch examples via tracker ─────────────────────────────")

    # Name-based fetch
    print(f"\n  [fetch_traces] name={TRACE_NAME!r}, limit=5:")
    try:
        results = tracker.fetch_traces(name=TRACE_NAME, limit=5)
        print(f"    → {len(results)} traces returned")
        for r in results[:2]:
            tid = r.get("id", "?") if isinstance(r, dict) else getattr(r, "id", "?")
            print(f"      {tid}")
    except Exception as exc:
        print(f"    → {exc}")

    # Tag-based fetch (Langfuse supports list-of-string tags)
    for tag in ["direct", "science", "hard"]:
        print(f"\n  [fetch_traces] tags=[{tag!r}], limit=3:")
        try:
            results = tracker.fetch_traces(tags=[tag], limit=3)
            print(f"    → {len(results)} traces")
        except Exception as exc:
            print(f"    → {exc}")

    # get_trace by ID
    sample_tid = trace_ids.get("direct", [""])[0]
    if sample_tid:
        print(f"\n  [get_trace] id={sample_tid[:8]}…:")
        try:
            trace = tracker.get_trace(sample_tid)
            name = trace.get("name", "?") if isinstance(trace, dict) else getattr(trace, "name", "?")
            print(f"    → name={name!r}")
        except Exception as exc:
            print(f"    → {exc}")


# ---------------------------------------------------------------------------
# Phase 5 – evaluate_traces (NotImplementedError — by design)
# ---------------------------------------------------------------------------

def show_evaluate_limitation(tracker) -> None:
    print("\n── Phase 5: evaluate_traces — Langfuse limitation ──────────────────")
    try:
        tracker.evaluate_traces(["trace-1"], scorers=["correctness"])
    except NotImplementedError as exc:
        print(f"  NotImplementedError (expected): {exc}")
    print(
        "\n  To run LLM-judge evaluation on Langfuse traces, use the Langfuse UI\n"
        "  or define a custom evaluator in the Langfuse dashboard.\n"
        "  For programmatic evaluation, switch to the MLflow backend:\n"
        "      tracker = create_tracker('mlflow', experiment_id='...')\n"
        "      tracker.evaluate_traces(trace_ids, scorers=[Correctness(...)])"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    use_mock = not os.getenv("ANTHROPIC_API_KEY") or os.getenv("MOCK_LLM") == "1"

    tracker = create_tracker(
        "langfuse",
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )

    categories = sorted({i["category"] for i in QA_DATASET})
    difficulties = {
        d: sum(1 for i in QA_DATASET if i["difficulty"] == d)
        for d in ["easy", "medium", "hard"]
    }

    print("Langfuse Tracing & Evaluation Demo  (via EvaluationTracker)")
    print(f"  Host          : {LANGFUSE_HOST}")
    print(f"  Dataset       : {DATASET_NAME}")
    print(f"  Variants      : {', '.join(VARIANTS)}")
    print(f"  Questions     : {len(QA_DATASET)}  ({', '.join(categories)})")
    print(f"  Difficulty    : {', '.join(f'{k}={v}' for k, v in difficulties.items())}")
    print(f"  Total traces  : {len(VARIANTS) * len(QA_DATASET)}")
    print(f"  LLM mode      : {'MOCK — set ANTHROPIC_API_KEY for real calls' if use_mock else 'Anthropic API'}")

    client = None
    if not use_mock:
        import anthropic
        client = anthropic.Anthropic()

    trace_ids = run_all_variants(tracker, use_mock=use_mock, client=client)
    create_eval_dataset(tracker)
    add_feedback(trace_ids, tracker)
    search_examples(trace_ids, tracker)
    show_evaluate_limitation(tracker)

    # Flush any buffered Langfuse events
    try:
        tracker._client.flush()
    except Exception:
        pass

    all_ids = [tid for ids in trace_ids.values() for tid in ids]
    print("\n── Done ────────────────────────────────────────────────────────────")
    print(f"  Logged {len(all_ids)} traces across {len(VARIANTS)} variants × {len(QA_DATASET)} questions")
    for variant, ids in trace_ids.items():
        print(f"    {variant:<22} {len(ids)} traces")
    print(f"\n  Explore: {LANGFUSE_HOST}")
    print()


if __name__ == "__main__":
    main()

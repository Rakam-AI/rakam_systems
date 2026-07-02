# Technical Spec — AI Gateway

- **Status:** Draft
- **Owner:** Roman
- **Created:** 2026-07-01
- **Scope doc (Notion):** _AI Gateway - rakam systems_ — https://app.notion.com/p/AI-Gateway-rakam-systems-39017b6f64ef80fbba7fd567605da3bc
- **Trigger:** Beetween onboarding (Azure in prod, Ollama/OpenAI-compatible in dev). Reported error `ValueError: Unknown provider: gemma4`.

This document is the engineering-level design. See the Notion scope doc for context and decisions; this spec assumes them.

---

## 1. Summary

Introduce a **thin, pydantic-ai-backed model factory** in `rakam_systems` that provides one standardized way to *define and resolve* both chat and embedding models, a Rakam-owned hook for usage metering, and an ingestion↔runtime embedding-consistency guarantee.

The gateway is a **factory**, not a call proxy: it returns configured `pydantic_ai` objects (a `Model` for chat, an `Embedder` for embeddings). Consumers use those objects directly, so pydantic-ai continues to own streaming, tool-calls, structured output, message history, and caching. Nothing is re-implemented.

---

## 2. Goals / non-goals

**Goals**
- One `provider:model` model-reference grammar for chat **and** embeddings.
- Full provider range via pydantic-ai; **no internal provider allow-list**.
- OpenAI-compatible `base_url` support (Azure / Ollama / local).
- Ingestion↔runtime embedding consistency (model + dimension) enforced, not conventional.
- A single seam for a private usage-metering hook (counts/metadata only).
- Copilot chat and the embedding services both obtain their models through the gateway.

**Non-goals**
- No network proxy / hosted gateway / sidecar.
- No custom rate limiting or caching (rely on provider SDK retries + pydantic-ai `FallbackModel` + provider-native caching).
- No reimplementation of provider SDKs.
- **No change to the existing raw-SDK `LLMGatewayFactory`** — it is still used by other consumers and coexists untouched.

---

## 3. Module layout

New package under the agents component (co-located with `BaseAgent`, which is the primary chat consumer):

```
ai-components/agents/src/rakam_systems_agent/components/model_gateway/
  __init__.py
  gateway.py          # ModelGateway: build_chat_model() / build_embedder()
  config.py           # ModelRef, GatewayConfig (pydantic models)
  metering.py         # UsageHook protocol + NoopUsageHook (impl stays private)
  consistency.py      # embedding model+dim record/validate helpers
```

The embedding-consistency helpers are consumed by the vector-store component; if a dependency direction issue arises, `consistency.py` moves to `rakam_systems_core`. Config schema types may live in `rakam_systems_core.config_schema` so both agents and vector-store share them (see §11).

---

## 4. Public API

```python
# config.py
class ModelRef(BaseModel):
    ref: str                      # "<provider>:<model>", e.g. "azure:gpt-4.1-mini"
    base_url: str | None = None   # optional OpenAI-compatible endpoint override
    # credentials resolved from the provider's standard env vars (not stored here)

class EmbeddingRef(ModelRef):
    dim: int                      # expected output dimension; validated against index metadata

# gateway.py
class ModelGateway:
    def __init__(self, usage_hook: UsageHook | None = None) -> None: ...

    def build_chat_model(self, cfg: ModelRef) -> pydantic_ai.models.Model:
        """Resolve cfg.ref via pydantic_ai infer_model, applying base_url and
        registering the usage hook. Returns a pydantic-ai Model for Agent(model=...)."""

    def build_embedder(self, cfg: EmbeddingRef) -> EmbeddingModel:
        """Resolve cfg.ref via pydantic_ai infer_embedding_model, applying base_url
        and registering the usage hook. Returns an adapter implementing the existing
        sync `rakam_systems_core.EmbeddingModel.run()` contract (vector stores are
        coded against it), bridging to pydantic-ai's async Embedder internally.
        For a local `sentence_transformer` ref, returns ConfigurableEmbeddings
        directly (no pydantic-ai — pydantic-ai has no local ST embedder)."""
```

**Resolution is passthrough** — the gateway never enumerates providers:

```python
from pydantic_ai.models import infer_model
from pydantic_ai.embeddings import infer_embedding_model, Embedder
```

`build_chat_model` wraps `infer_model(cfg.ref)`; when `cfg.base_url` is set it constructs the provider explicitly (`OpenAIProvider(base_url=...)`) and passes it through. `build_embedder` mirrors this with `infer_embedding_model`, then wraps the resulting async `Embedder` in a sync `EmbeddingModel` adapter (see §8a) so the vector stores stay unchanged.

> **Embedder return type is deliberately `EmbeddingModel`, not `pydantic_ai.Embedder`.** The stores call sync `EmbeddingModel.run(texts) -> List[List[float]]`; pydantic-ai's `Embedder` exposes async `embed_documents`/`embed_query`. Returning the raw Embedder would break every store. The adapter bridges async→sync (dedicated event loop / `asyncio.run` at the call boundary — validate under the services' existing async context).

> Guardrail: the only mapping the gateway owns is `ref → pydantic-ai object`. It must not maintain a `_PROVIDERS` dict. An unknown provider surfaces pydantic-ai's own error unchanged.

---

## 5. Configuration

The `ref` grammar is identical for chat and embeddings; credentials/endpoint come from standard provider env vars. **Back-compat: reuse the existing `model` string as `ref`** — existing client configs (e.g. `model: "openai-responses:gpt-5.2"`) keep working unchanged; `base_url`/endpoint is an *additive* field only for clients on a custom endpoint. Example per-deployment config:

```yaml
# chat
model:
  ref: "azure:gpt-4.1-mini"
# env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION

# embeddings
embedding:
  ref: "azure:text-embedding-3-small"
  dim: 1536
```

Dev:

```yaml
model:     { ref: "ollama:gemma2" }                      # OLLAMA_BASE_URL
embedding: { ref: "openai:nomic-embed-text",             # base_url → Ollama /v1
             base_url: "http://localhost:11434/v1", dim: 768 }
```

Provider → env reference:

| Target | `ref` | Env |
|---|---|---|
| Azure | `azure:<deployment>` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION` |
| Ollama | `ollama:<model>` | `OLLAMA_BASE_URL` |
| OpenAI-compatible | `openai:<model>` + `base_url` | `OPENAI_API_KEY` (dummy allowed) |
| OpenAI | `openai:<model>` | `OPENAI_API_KEY` |

---

## 6. Usage metering hook

```python
# metering.py
class UsageHook(Protocol):
    def record(self, *, ref: str, kind: Literal["chat", "embedding"],
               usage: Usage, latency_ms: float) -> None: ...

class NoopUsageHook:
    def record(self, **_: object) -> None: ...
```

- The **interface** ships in OSS `rakam_systems`; the default is `NoopUsageHook`.
- The **Rakam AI APIs metering implementation is private** (copilot config or a private module), injected via `ModelGateway(usage_hook=...)`.
- `record` receives **counts/metadata only** (`usage` = token counts from pydantic-ai's per-call usage, model ref, latency). **Never prompt/response content** — required for per-client data residency.
- The gateway attaches the hook so it fires after each chat/embedding call using pydantic-ai's reported usage; it does not compute cost (existing `genai-prices` continues to own that).

---

## 7. Ingestion ↔ runtime embedding consistency

Problem: the RAG index / graph is embedded once at ingestion; query-time must use the same model + dimension or retrieval degrades silently.

Mechanism:

1. **At ingestion**, after building the embedder, record `{ref, dim}` into the index/collection metadata (`consistency.record(collection, cfg)`).
2. **At runtime startup**, `consistency.validate(collection, cfg)` reads the stored `{ref, dim}` and compares against the configured `EmbeddingRef`. On mismatch → raise a clear startup error naming both sides.
3. Provider-only change with an identical model (e.g. `text-embedding-3-small` on Azure vs OpenAI → identical vectors) is **compatible** — validation keys on `model` + `dim`, not provider. A different model/dimension requires corpus regeneration.

Where stored: pgvector collection metadata for ticket-rag; graph store metadata for the graph. Exact column/property TBD with the vector-store owner (§11).

---

## 8. Consumer integration

| Consumer | Change | File(s) |
|---|---|---|
| Copilot chat | **Deferred** — no near-term payoff without metering, and it re-plumbs a working path. Revisit with the metering feature spec: `BaseAgent`/`_resolve_model` would obtain the model from `ModelGateway.build_chat_model(...)` and the private metering hook would be injected in the copilot. | `ots-copilot-agent` `services/agent/agent.py`, `rakam_systems_agent` `base_agent.py` |
| Graph embedder | Build via `build_embedder(...)`; add `record`/`validate`. | `ots-graph-generation-service` `shared/embedding/encoder.py` |
| Ticket-RAG | Build via `build_embedder(...)`; add `record`/`validate`. | `ots-ticket-service` `services/ticket-rag/.../vector_store.py` |
| Raw-SDK `LLMGatewayFactory` | **Unchanged** — coexists. | `rakam_systems_agent/components/llm_gateway/*` |

The Bedrock profile handling currently in `_resolve_model` moves behind `build_chat_model` (still special-cased there), so consumers stay uniform.

---

## 9. Coexistence with `LLMGatewayFactory`

The existing raw-SDK gateway (`openai`/`mistral`) remains in place and is not modified. The new `ModelGateway` is additive and used by new provider needs (Azure, Ollama, embeddings, copilot routing). No shared state; the two can live side by side indefinitely. A future consolidation is out of scope for this spec.

---

## 10. Testing

- **Unit:** `build_chat_model` / `build_embedder` resolve known `ref`s to the correct pydantic-ai model type; `base_url` is threaded to the provider; an unknown provider raises pydantic-ai's error unchanged (no swallowing).
- **Unit:** `UsageHook.record` is invoked once per call with token counts and never content; `NoopUsageHook` default is a no-op.
- **Unit:** consistency `validate` passes on match, raises on model/dim mismatch, and treats same-model/different-provider as compatible.
- **Integration (critical path, real deps preferred):** an `Agent` built from `build_chat_model("openai:...")` against a live/OpenAI-compatible endpoint; an `Embedder` from `build_embedder(...)` producing vectors of the declared `dim`. Prefer a local Ollama/OpenAI-compatible endpoint in CI over mocks.

---

## 11. Open questions

1. ~~Config schema home~~ — **decided:** `ModelRef`/`EmbeddingRef` live in `rakam_systems_core.config_schema` so agents and vector-store share them without depending on each other.
2. ~~`build_embedder` vs `ConfigurableEmbeddings`~~ — **decided:** wrap, not replace. `build_embedder` returns an `EmbeddingModel` adapter over pydantic-ai for API providers, and returns `ConfigurableEmbeddings` directly for the local `sentence_transformer` backend (retained; pydantic-ai has no local ST equivalent).
3. ~~Consistency metadata storage~~ — **decided:** record `{model, dim}` on the existing per-index record (ticket-rag `ticket_indices` table) and as a metadata property on the graph store; validate at startup.
4. Graph 384-dim truncation — **decided:** keep `dim` as its own config field and pass it to the provider's `dimensions` parameter; `ref` (model name) is unchanged.
5. Metering hook transport + auth to the Rakam AI APIs collection — out of scope here; covered by the separate metering feature spec.

---

## 12. Milestones

- **M0 (Beetween unblock, no gateway):** copilot chat→Azure via config/env; quick `base_url` on the embedder; keep `text-embedding-3-small` for test so the existing corpus stays valid.
- **M1:** `ModelGateway` + config schema + `build_chat_model`/`build_embedder` + tests (passthrough, no allow-list).
- **M2:** consistency record/validate; wire graph + ticket-rag embedders.
- **M3:** copilot routes chat through the gateway; inject metering hook (private).

---

## 13. Implementation plan (PR breakdown)

PR-sized, reviewable units. Order reflects dependencies; PRs in different repos can proceed in parallel once their `rakam_systems` dependency is released.

| PR | Repo | Scope | Depends on | Notes |
|---|---|---|---|---|
| **PR 0 — Beetween unblock** | `beetween-support-agent-config` | Chat → Azure: `model` ref + `AZURE_*` env; embedder `base_url` for test | — | Config only, no gateway. Ships independently of everything below. |
| **PR 1 — config types** | `rakam_systems` (core) | `ModelRef` / `EmbeddingRef` in `rakam_systems_core.config_schema` + unit tests | — | Foundational, tiny. |
| **PR 2 — chat factory** | `rakam_systems` (agents) | `model_gateway` module; `build_chat_model` (passthrough to `infer_model`, `base_url`, Bedrock-profile move); `UsageHook` protocol + `NoopUsageHook` | PR 1 | Tests: resolution, **no allow-list**, `base_url` threading, unknown-provider surfaces pydantic-ai's error unchanged. |
| **PR 3 — embed factory + adapter** | `rakam_systems` (vector-store) | `build_embedder` → sync `EmbeddingModel` adapter over async `Embedder`; local `sentence_transformer` passthrough | PR 1, 2 | Prototype the **async→sync bridge** first — main implementation risk. |
| **PR 4 — consistency helpers** | `rakam_systems` (core/vector-store) | `record` / `validate` of `{model, dim}`; storage contract | PR 3 | Startup-fail on mismatch. |
| **PR 5 — wire ticket-rag** | `ots-ticket-service` | `_build_embeddings` via `build_embedder`; record/validate on `ticket_indices`; **+pydantic-ai (pinned)** | PR 3, 4 | Integration test against a live/OpenAI-compatible endpoint. |
| **PR 6 — wire graph** | `ots-graph-generation-service` | `encoder.py` via `build_embedder`; record/validate; 384-dim via `dimensions` param; **+pydantic-ai (pinned)** | PR 3, 4 | Integration test. |
| **PR 7 — ingestion consistency** | `rs-data-ingestion` | Ensure orchestrated ingest drives the same `ref`/`dim` as query time | PR 5 | Guards ingestion↔runtime match end-to-end. |
| **Deferred — metering + copilot routing** | private + `ots-copilot-agent` | `UsageHook` impl to Rakam AI APIs; route copilot chat through gateway | PR 2 | Separate feature spec. |

**Critical path:** PR 1 → PR 2 → PR 3 → PR 4, then PR 5 / PR 6 in parallel, then PR 7. PR 0 is independent and unblocks Beetween immediately.

**CI note:** PR 5 and PR 6 add `pydantic-ai` to services that don't have it — pin it in `pyproject` (our CI resolves fresh and ignores `uv.lock`, so open bounds drift).

# Tech spec — `rakam-systems-documents` 0.2.0 (F2c.1)

Implementation-level companion to
[documents-page-segmentation.md](./documents-page-segmentation.md).

**Branch:** `dev_w32` (off `origin/main` @ `0da09c9`)
**Package root:** `documents/src/rakam_systems_documents/`

---

## Contract added

A page delimiter, emitted by **every** paged branch so consumers never have to
ask which engine produced a document before slicing it:

```
<!-- page:1 -->
…page 1 markdown…

<!-- page:2 -->
…
```

Exported as a module constant so the agent slices on the same literal rather
than duplicating a regex:

```python
# schema.py
PAGE_DELIMITER = "<!-- page:{n} -->"
PAGE_DELIMITER_RE = re.compile(r"^<!-- page:(\d+) -->$", re.MULTILINE)
```

Add both to `__init__.__all__`. An HTML comment is used because it renders as
nothing in any markdown viewer, cannot collide with document prose, and does
not perturb a model reading the text.

---

## `prepare.py` changes

### `_prepare_pdf` — native-text branch

```python
chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)   # list[dict]
```

`page_chunks=True` is confirmed present in the pinned `pymupdf4llm` 0.0.17
(verified against the installed signature; the parameter list also carries
`extract_words`, unused here). Each chunk is a dict with `text` and a
`metadata` mapping carrying the page number.

Join as `PAGE_DELIMITER.format(n=…) + "\n" + chunk_text`, separated by blank
lines. Read the page number from chunk metadata rather than enumerating — do
not assume the library returns chunks in page order for every document.

Keep `provenance` as one `SourceRef(page=n)` per page. Shape is unchanged; the
difference is that each entry now has a locatable region in `markdown`.

### `_prepare_pdf` — OCR branch, and `_prepare_image`

`MistralOCRProvider.ocr()` currently joins page markdown with `"\n\n"` and
returns `[SourceRef(page=i+1)]`. Move the delimiter emission **into the
providers** so `DoclingOCRProvider` and any future provider are held to the same
contract by construction, rather than each caller re-deriving it.

`_prepare_image` yields a single page — emit `<!-- page:1 -->` rather than
special-casing, so consumers have exactly one code path.

### Truncation telemetry

In `prepare()`, before the `max_chars` cut:

```python
pc.meta["total_chars"] = len(pc.markdown)
```

Set unconditionally, so a consumer can always render *"showing 8000 of
143000"*. `truncated` keeps its current meaning.

**Truncation must not sever a page mid-delimiter.** Cutting at `max_chars` can
land inside `<!-- page:12 -->` and leave a fragment that breaks the consumer's
regex. Truncate to the last **complete** page boundary at or before
`max_chars`; if even page 1 exceeds the limit, cut mid-page but never mid-
delimiter. Record the last whole page in `meta["truncated_after_page"]`.

Extend `meta["page_count"]` to the OCR branch (currently native-only).

---

## Non-goals

No bounding boxes (W32 decision D6). No chunking, embedding or retrieval — this
package stays a pure `bytes -> PreparedContent` primitive. No new dependencies:
`page_chunks` is a parameter on an already-pinned library, and the
`pymupdf4llm>=0.0.17,<0.0.18` cap stays (0.0.18 pulls `onnxruntime`, which has
no cp310 wheels).

---

## Compatibility

`PreparedContent`'s field set is unchanged — `markdown` gains delimiters,
`meta` gains keys. Consumers treating `markdown` as opaque text are unaffected.

Documents prepared by 0.1.0 carry no delimiters. **The degradation rule is the
consumer's**, specified agent-side in F2c.3: no delimiters → treat as one page,
ignore `page_range`, never error.

---

## Tests

`documents/src/rakam_systems_documents/tests/test_prepare.py`:

| Case | Assertion |
|---|---|
| 3-page native PDF | 3 delimiters, ascending; page *n*'s text between delimiter *n* and *n+1* |
| 1-page native PDF | exactly one delimiter (no special-casing) |
| Scanned PDF, stub `OCRProvider` | identical delimiter contract to native |
| Image | single `page:1` delimiter |
| `total_chars` | equals pre-truncation length; present even when not truncated |
| Truncation at a page boundary | never cuts inside a delimiter; `truncated_after_page` set |
| `max_chars` smaller than page 1 | cuts mid-page, delimiter intact |
| XLSX / CSV | unchanged; `rows` provenance untouched |
| Unreadable bytes | still `provider="none"` + `meta["help"]`, never raises |

Run on **3.11** — the repo `.venv` is 3.9-era and cannot evaluate `int | None`
annotations in this package (hit during W30; use a `uv` 3.11 scratch venv).

---

## Release

`documents/pyproject.toml` 0.1.0 → **0.2.0**.

Publish target is **PyPI**, not GitHub releases — there is no release tag to
look for, so `gh release list` will not show it. The agent then repins
`rakam-systems-documents==0.2.0` and re-locks.

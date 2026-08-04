# F2c.1 — Page-segmented prepare (`rakam-systems-documents` 0.2.0)

**Status:** spec · **Date:** 2026-08-03 · **Branch:** `dev_w32`

Part of [W32 — Document analysis](https://github.com/Rakam-AI/ots-copilot-agent/blob/dev_w32/docs/W32-architectural-review/README.md).
This is the **first** item in the chain: the agent repins this package, then the
portal consumes what the agent exposes.

---

## Problem

`PreparedContent.provenance` currently claims to trace fragments to their source
but, for native PDFs, is a bare list of every page number:

```python
# prepare.py, _prepare_pdf — native-text branch
markdown = pymupdf4llm.to_markdown(doc)          # one flat string, pages fused
provenance=[SourceRef(page=i + 1) for i in range(pages)]
```

Nothing links a span of markdown to the page it came from. So a consumer can
report *"this document has 12 pages"* but never *"this claim is on page 7"* —
which is exactly what W32's file citations (F2c.4) and page-scoped reads (F2c.3)
need.

Spreadsheets are already correct: `TableRow.source` carries real `(sheet, row)`.
It is only the PDF/prose path that loses the mapping.

Separately, truncation is invisible. `prepare()` cuts `markdown` at `max_chars`
and sets `truncated=True`, but never records how much there was, so no caller
can tell a 1% trim from a 90% one.

---

## Change

### 1. Page-segmented native-PDF extraction

`pymupdf4llm` 0.0.17 accepts `page_chunks=True`, returning per-page chunks
instead of one fused string (verified in the installed package; the parameter
list also includes `extract_words`).

Emit page-delimited markdown and provenance that actually corresponds to it.
The delimiter must be stable and machine-parseable, because F2c.3 slices on it
to serve `page_range`:

```
<!-- page:1 -->
…markdown for page 1…

<!-- page:2 -->
…
```

An HTML comment is chosen over a heading so it renders as nothing in any
markdown viewer, cannot collide with document content, and does not perturb a
model reading the text.

`provenance` keeps one `SourceRef` per page, unchanged in shape — but now every
entry has a corresponding, locatable region in `markdown`.

### 2. Truncation telemetry

Add to `meta`:

- `total_chars` — length **before** truncation
- `page_count` — already present for PDFs; extend to the OCR branch

`truncated` keeps its current meaning. Consumers can now show *"showing 8000 of
143000 characters"* instead of silently lying by omission.

### 3. OCR branch

`MistralOCRProvider.ocr()` already returns one `SourceRef` per page and joins
page markdown with `\n\n`. Apply the same page delimiter there so both branches
produce the identical contract, and a consumer never has to ask which engine
produced a document before it can slice it.

---

## Non-goals

- **No bounding boxes.** Out of scope for W32 (decision D6). Mistral's OCR
  response carries bboxes only on *extracted images*, never on text spans, so
  page-level is the ceiling for scans regardless of effort spent here.
  `extract_words=True` remains available for a future native-PDF-only exact
  highlight, without an engine change.
- **No chunking, embedding, or retrieval.** This package stays a pure,
  side-effect-free primitive: bytes in, `PreparedContent` out. Chunking policy
  belongs to the consumer (decision D2 rejects session RAG outright).
- **No new dependencies.** `page_chunks` is a parameter on a library already
  pinned; the `pymupdf4llm>=0.0.17,<0.0.18` cap stays (0.0.18 pulls
  `onnxruntime`, which has no cp310 wheels and defeats the light-install goal).

---

## Compatibility

`PreparedContent`'s field set is unchanged — only `markdown` gains delimiters
and `meta` gains keys. Consumers that treat `markdown` as opaque text are
unaffected; the delimiter is an HTML comment and renders as nothing.

The agent stores the full dump in `session_files.prepared_content` (JSON), so
no migration is needed anywhere downstream.

Documents prepared by 0.1.0 have no delimiters. Consumers must degrade to
whole-document behaviour rather than error when they find none — specified on
the agent side in F2c.3.

Version **0.1.0 → 0.2.0**. Publish target is **PyPI**, not GitHub releases —
there is no release tag to look for (`gh release list` will not show it).

---

## Tests

Extend `documents/src/rakam_systems_documents/tests/test_prepare.py`:

- Multi-page native PDF → one delimiter per page; page *n*'s text sits between
  delimiter *n* and *n+1*.
- Single-page PDF → exactly one delimiter (no special-casing).
- `total_chars` reflects pre-truncation length; `truncated` true only when cut.
- Scanned-PDF path (stub `OCRProvider`) → same delimiter contract as native.
- Spreadsheet/CSV path → unchanged; `rows` provenance untouched.
- A 0.1.0-shaped document (no delimiters) still parses as one page.

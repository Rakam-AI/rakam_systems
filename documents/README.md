# rakam-systems-documents

Generic **document preparation**: turn any source document into clean, structured,
provenance-tagged content the rest of the stack can reason over.

```python
from rakam_systems_documents import prepare, MistralOCRProvider

pc = prepare(raw_bytes, mime="application/pdf", filename="invoice.pdf",
             ocr=MistralOCRProvider())      # ocr optional; native PDFs skip it
pc.markdown      # what a model reads
pc.rows          # structured rows (tabular sources), header-keyed + provenance
pc.provenance    # page/sheet/row refs
pc.provider      # pdf_text | mistral_ocr | docling_ocr | tabular | text | none
```

## What it is (and isn't)

**One job:** `bytes -> PreparedContent`. Format dispatch, parse/decode, hybrid-OCR
routing, provenance normalization, truncation. A pure function — the only I/O is the
injected OCR provider.

**Not** its job: storage, chunking, embedding, indexing, async/status, or any business
meaning (extraction, matching, classification). Those belong to the caller — the library
gives every caller one document→content primitive so none reimplements parsing/OCR.

## Formats

| Source | Handling |
|---|---|
| Native PDF | `pymupdf4llm` → markdown (text layer, exact, no egress) |
| Scanned PDF / image | **hybrid gate** → OCR provider (see below) |
| Excel / CSV | `openpyxl` / `csv` → markdown table **+ structured rows + (sheet,row) provenance** |
| Email / text | rfc822 + multi-encoding decode (binary rejected) |

## OCR is pluggable

Native-text PDFs never hit OCR. Scanned pages / images route to an `OCRProvider`:

- **`MistralOCRProvider`** (default) — the Mistral hosted OCR API (`/v1/ocr`). Needs `MISTRAL_API_KEY`.
- **`DoclingOCRProvider`** — on-prem, no egress. Requires the optional extra:
  `pip install rakam-systems-documents[docling]`.

A missing/unavailable engine degrades gracefully (native text still works; a scan yields
`provider="none"` + a help hint, never an exception).

## Install

```
pip install rakam-systems-documents            # Mistral OCR path
pip install rakam-systems-documents[docling]    # + on-prem OCR
```

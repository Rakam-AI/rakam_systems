"""The neutral currency the prepare core moves: ``PreparedContent``.

Downstream-agnostic on purpose — it names *content* (markdown, structured rows,
provenance), never business meaning. Extraction / matching / classification are
the consumer's job, not this package's.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """Where a fragment came from. Best-effort per provider: exact for native
    PDFs / spreadsheets, page-level for scan-OCR, empty when unknown."""

    page: int | None = None      # PDF / scan (1-indexed)
    sheet: str | None = None     # spreadsheet
    row: int | None = None       # spreadsheet / table (1-indexed)


class TableRow(BaseModel):
    """One row of a tabular source, header-keyed when a header is detected."""

    cells: dict[str, str]
    source: SourceRef = Field(default_factory=SourceRef)


class PreparedContent(BaseModel):
    """The result of preparing one document.

    ``markdown`` is what a model reads; ``rows`` carries structured tabular data
    for precise downstream mapping; ``provenance`` traces prose/scan fragments to
    their source. ``meta`` holds light, non-business signals only
    (page_count / sheet_names / encoding / help)."""

    markdown: str = ""
    rows: list[TableRow] = Field(default_factory=list)
    provenance: list[SourceRef] = Field(default_factory=list)
    provider: str = "none"        # pdf_text | mistral_ocr | docling_ocr | tabular | text | none
    truncated: bool = False
    meta: dict = Field(default_factory=dict)

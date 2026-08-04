"""The neutral currency the prepare core moves: ``PreparedContent``.

Downstream-agnostic on purpose — it names *content* (markdown, structured rows,
provenance), never business meaning. Extraction / matching / classification are
the consumer's job, not this package's.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field


# Page boundary marker embedded in ``markdown`` by every paged branch (native
# PDF, OCR, image), so a consumer can slice by page without knowing which
# engine produced the document.
#
# An HTML comment rather than a heading: it renders as nothing in any markdown
# viewer, does not perturb a model reading the text, and cannot be confused
# with document prose. pymupdf4llm's own inter-page rule is a bare ``-----``,
# which a document containing a horizontal rule is indistinguishable from —
# that ambiguity is exactly what this replaces.
PAGE_DELIMITER = "<!-- page:{n} -->"
PAGE_DELIMITER_RE = re.compile(r"^<!-- page:(\d+) -->$", re.MULTILINE)


def page_delimiter(n: int) -> str:
    """The marker introducing page *n* (1-indexed)."""
    return PAGE_DELIMITER.format(n=n)


def split_pages(markdown: str) -> list[tuple[int, str]]:
    """Split delimited markdown into ``(page_number, text)`` pairs.

    Degrades deliberately: content with no delimiters — anything prepared by
    0.1.x, or a non-paged source such as a spreadsheet — comes back as a single
    ``(1, markdown)`` pair rather than raising, so callers need no version
    check."""
    matches = list(PAGE_DELIMITER_RE.finditer(markdown))
    if not matches:
        return [(1, markdown)] if markdown else []
    pages: list[tuple[int, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        pages.append((int(match.group(1)), markdown[start:end].strip("\n")))
    return pages


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

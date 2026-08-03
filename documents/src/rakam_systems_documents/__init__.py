"""rakam-systems-documents — generic document preparation.

``prepare(bytes, mime, filename) -> PreparedContent``: any source (native/scanned
PDF, image, Excel/CSV, email/text) to markdown + structured rows + provenance.
A pure, side-effect-free primitive; storage / chunking / embedding / extraction
belong to the caller.
"""
from __future__ import annotations

from .prepare import prepare
from .providers import DoclingOCRProvider, MistralOCRProvider, OCRProvider
from .schema import (
    PAGE_DELIMITER,
    PAGE_DELIMITER_RE,
    PreparedContent,
    SourceRef,
    TableRow,
    page_delimiter,
    split_pages,
)

__version__ = "0.2.0"

__all__ = [
    "prepare",
    "PreparedContent",
    "TableRow",
    "SourceRef",
    "OCRProvider",
    "MistralOCRProvider",
    "DoclingOCRProvider",
    # Page segmentation (0.2.0) — consumers slice on the shared constant
    # rather than re-deriving the marker.
    "PAGE_DELIMITER",
    "PAGE_DELIMITER_RE",
    "page_delimiter",
    "split_pages",
]

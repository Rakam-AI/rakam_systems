"""rakam-systems-documents — generic document preparation.

``prepare(bytes, mime, filename) -> PreparedContent``: any source (native/scanned
PDF, image, Excel/CSV, email/text) to markdown + structured rows + provenance.
A pure, side-effect-free primitive; storage / chunking / embedding / extraction
belong to the caller.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

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

# Read from the installed distribution rather than restated here. The release
# workflow bumps `pyproject.toml` only, so a hand-written literal drifts on the
# very next release: this one said "0.2.0" while the published package was
# 0.1.1 — a version that never existed on PyPI. pyproject is the single source
# of truth.
try:
    __version__ = _metadata_version("rakam-systems-documents")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "prepare",
    "PreparedContent",
    "TableRow",
    "SourceRef",
    "OCRProvider",
    "MistralOCRProvider",
    "DoclingOCRProvider",
    # Page segmentation — consumers slice on the shared constant rather than
    # re-deriving the marker.
    "PAGE_DELIMITER",
    "PAGE_DELIMITER_RE",
    "page_delimiter",
    "split_pages",
]

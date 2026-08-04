"""OCR for scanned pages / images — behind one interface so the engine is a
config choice, never a fork. Mistral is the default (hosted API); Docling is the
on-prem, no-egress swap (optional extra ``rakam-systems-documents[docling]``).
"""
from __future__ import annotations

import base64
import os
from typing import Protocol, runtime_checkable

from ..schema import SourceRef, page_delimiter


@runtime_checkable
class OCRProvider(Protocol):
    """An OCR engine. ``ocr`` returns ``(markdown, provenance)`` and must be
    best-effort — the caller treats an empty return as "unreadable", never an
    exception path.

    A provider that can attribute text to pages MUST delimit its markdown with
    :func:`~rakam_systems_documents.schema.page_delimiter`, so consumers slice
    OCR output exactly as they slice native-PDF output. One that cannot returns
    undelimited markdown and ``split_pages`` degrades it to a single page —
    correct, just coarser."""

    name: str

    def available(self) -> bool: ...

    def ocr(self, content: bytes, mime: str) -> tuple[str, list[SourceRef]]: ...


class MistralOCRProvider:
    """Default engine — the Mistral hosted OCR API (``/v1/ocr``)."""

    name = "mistral"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-ocr-latest",
        timeout: float = 90.0,
    ) -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = os.getenv("OCR_MODEL", model)
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def ocr(self, content: bytes, mime: str) -> tuple[str, list[SourceRef]]:
        import httpx

        is_pdf = mime == "application/pdf" or content[:5] == b"%PDF-"
        field = "document_url" if is_pdf else "image_url"
        data_url = f"data:{mime};base64,{base64.b64encode(content).decode()}"
        payload = {"model": self.model, "document": {"type": field, field: data_url}}
        resp = httpx.post(
            "https://api.mistral.ai/v1/ocr",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        pages = resp.json().get("pages", [])
        # The API returns one markdown blob per page, so real page boundaries
        # are available — emit them rather than fusing the pages together.
        # (Note: the response carries bounding boxes only for *extracted
        # images*, never for text spans, so page level is the finest
        # attribution this engine can support.)
        markdown = "\n\n".join(
            f"{page_delimiter(i)}\n{p.get('markdown', '')}"
            for i, p in enumerate(pages, start=1)
        )
        return markdown, [SourceRef(page=i + 1) for i in range(len(pages))]


class DoclingOCRProvider:
    """On-prem, no-egress OCR via Docling. Requires the ``docling`` extra; when
    absent, ``available()`` is False and the caller falls back to text-only."""

    name = "docling"

    def available(self) -> bool:
        try:
            import docling  # noqa: F401

            return True
        except ImportError:
            return False

    def ocr(self, content: bytes, mime: str) -> tuple[str, list[SourceRef]]:
        import os as _os
        import tempfile

        from docling.document_converter import DocumentConverter

        is_pdf = mime == "application/pdf" or content[:5] == b"%PDF-"
        suffix = ".pdf" if is_pdf else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(content)
            path = tf.name
        try:
            document = DocumentConverter().convert(path).document
            markdown = document.export_to_markdown()
        finally:
            _os.unlink(path)
        # ``export_to_markdown`` fuses the document into one string, so unlike
        # the Mistral path there is no per-page text to delimit. Returned
        # undelimited on purpose: ``split_pages`` degrades it to a single page,
        # which is coarse but honest. Fabricating a ``page:1`` marker for a
        # multi-page scan would attribute every citation to page 1.
        pages = getattr(document, "num_pages", None) or 1
        return markdown, [SourceRef(page=i + 1) for i in range(pages)]

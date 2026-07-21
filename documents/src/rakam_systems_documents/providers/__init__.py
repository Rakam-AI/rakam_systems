"""OCR providers — the pluggable, engine-agnostic seam for scanned documents."""
from __future__ import annotations

from .ocr import DoclingOCRProvider, MistralOCRProvider, OCRProvider

__all__ = ["OCRProvider", "MistralOCRProvider", "DoclingOCRProvider"]

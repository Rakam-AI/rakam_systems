"""Behavioural tests for prepare() — one per dispatch/route, using in-test
sample bytes (native PDF, scanned PDF, xlsx, csv, email, binary).
"""
from __future__ import annotations

import io
from email.message import EmailMessage

import fitz  # pymupdf
import openpyxl
import pytest

from rakam_systems_documents import PreparedContent, SourceRef, prepare

PDF = "application/pdf"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ── sample builders ─────────────────────────────────────────────────────────
def _native_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "OFFRE FOURNISSEUR ACME\nTube acier E24 50x50\n1200 ml", fontsize=11)
    return doc.tobytes()


def _scanned_pdf() -> bytes:
    src = fitz.open()
    src.new_page().insert_text((72, 72), "FACTURE scan\nTotal 5040 EUR", fontsize=14)
    pix = src[0].get_pixmap(dpi=120)
    out = fitz.open()
    page = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
    page.insert_image(page.rect, pixmap=pix)  # image only → no text layer
    return out.tobytes()


def _catalogue_xlsx() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Catalogue"
    ws.append(["Code", "Designation", "Grade"])
    ws.append(["ST-E24-5050", "Tube acier 50x50x3", "E24"])
    ws.append(["ST-S235-6060", "Tube acier 60x60x4", "S235"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _email() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Demande de prix"
    msg["From"] = "achat@client.example"
    msg.set_content("Merci de chiffrer 1200 ml de tube acier E24.")
    return bytes(msg)


class _MockOCR:
    name = "mistral"

    def available(self) -> bool:
        return True

    def ocr(self, content: bytes, mime: str):
        return "# (OCR) FACTURE\n\nTotal 5040 EUR", [SourceRef(page=1)]


# ── tests ────────────────────────────────────────────────────────────────────
def test_native_pdf_uses_text_layer_not_ocr():
    called = {"ocr": False}

    class _Spy(_MockOCR):
        def ocr(self, content, mime):
            called["ocr"] = True
            return "should-not-be-used", []

    pc = prepare(_native_pdf(), PDF, "offer.pdf", ocr=_Spy())
    assert pc.provider == "pdf_text"
    assert called["ocr"] is False
    assert "ACME" in pc.markdown
    assert pc.provenance and pc.provenance[0].page == 1


def test_scanned_pdf_routes_to_ocr():
    pc = prepare(_scanned_pdf(), PDF, "invoice.pdf", ocr=_MockOCR())
    assert pc.provider == "mistral_ocr"
    assert "5040" in pc.markdown


def test_scanned_pdf_without_ocr_degrades_gracefully():
    pc = prepare(_scanned_pdf(), PDF, "invoice.pdf", ocr=None)
    assert pc.provider == "none"
    assert "help" in pc.meta
    assert pc.markdown == ""


def test_xlsx_emits_markdown_table_and_structured_rows_with_provenance():
    pc = prepare(_catalogue_xlsx(), XLSX, "catalogue.xlsx")
    assert pc.provider == "tabular"
    assert "| Code | Designation | Grade |" in pc.markdown
    assert len(pc.rows) == 2
    first = pc.rows[0]
    assert first.cells["Code"] == "ST-E24-5050"
    assert first.cells["Grade"] == "E24"
    assert first.source.sheet == "Catalogue"
    assert first.source.row == 2


def test_csv_emits_rows_with_row_provenance():
    csv_bytes = b"produit,quantite\nTube E24,1200\nEPI,30\n"
    pc = prepare(csv_bytes, "text/csv", "req.csv")
    assert pc.provider == "tabular"
    assert len(pc.rows) == 2
    assert pc.rows[0].cells == {"produit": "Tube E24", "quantite": "1200"}
    assert pc.rows[0].source.row == 2


def test_email_decodes_body_and_keeps_subject():
    pc = prepare(_email(), "message/rfc822", "req.eml")
    assert pc.provider == "text"
    assert pc.meta["subject"] == "Demande de prix"
    assert "tube acier E24" in pc.markdown


def test_binary_masquerading_as_text_is_rejected():
    pc = prepare(b"\x89PNG\r\n\x00\x00garbage", "text/plain", "fake.txt")
    assert pc.provider == "none"
    assert "help" in pc.meta


def test_plain_text_multi_encoding():
    # latin-1-encodable accents (em-dash is not latin-1) — exercises the fallback decode
    pc = prepare("café déjà vu à Genève".encode("latin-1"), "text/plain", "note.txt")
    assert pc.provider == "text"
    assert "café" in pc.markdown


def test_markdown_truncated_to_max_chars():
    big = ("# title\n" + "x" * 20000).encode()
    pc = prepare(big, "text/markdown", "big.md", max_chars=100)
    assert pc.truncated is True
    assert len(pc.markdown) == 100


def test_prepare_never_raises_returns_prepared_content():
    pc = prepare(b"", "application/octet-stream", "empty.bin")
    assert isinstance(pc, PreparedContent)
    assert pc.provider == "none"

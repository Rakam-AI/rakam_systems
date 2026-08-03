"""Behavioural tests for prepare() — one per dispatch/route, using in-test
sample bytes (native PDF, scanned PDF, xlsx, csv, email, binary).
"""
from __future__ import annotations

import io
from email.message import EmailMessage

import fitz  # pymupdf
import openpyxl

from rakam_systems_documents import (
    PAGE_DELIMITER_RE,
    PreparedContent,
    SourceRef,
    page_delimiter,
    prepare,
    split_pages,
)

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


# ── page segmentation (0.2.0) ───────────────────────────────────────────────
def _multipage_pdf(n: int = 3) -> bytes:
    doc = fitz.open()
    for i in range(1, n + 1):
        doc.new_page().insert_text(
            (72, 72), f"PAGE {i} CONTENT\nligne {i} unique marker Z{i}Z", fontsize=11,
        )
    return doc.tobytes()


def test_native_pdf_emits_one_delimiter_per_page_in_order():
    pc = prepare(_multipage_pdf(3), PDF, "multi.pdf", max_chars=100000)
    pages = split_pages(pc.markdown)

    assert [n for n, _ in pages] == [1, 2, 3]
    # Each page's own marker text lands under its own delimiter — the property
    # that makes a file+page citation meaningful rather than decorative.
    for n, text in pages:
        assert f"Z{n}Z" in text
        assert f"Z{n + 1}Z" not in text


def test_single_page_pdf_is_not_special_cased():
    pc = prepare(_native_pdf(), PDF, "one.pdf", max_chars=100000)
    assert len(split_pages(pc.markdown)) == 1
    assert PAGE_DELIMITER_RE.search(pc.markdown) is not None


def test_pymupdf_own_page_rule_is_stripped():
    """pymupdf4llm ends each page with a bare '-----'. Left in, a document
    would carry two competing page markers, one of them ambiguous with any
    genuine horizontal rule in the source."""
    pc = prepare(_multipage_pdf(2), PDF, "multi.pdf", max_chars=100000)
    assert "-----" not in pc.markdown


def test_ocr_path_uses_the_same_delimiter_contract():
    """A consumer must not need to know which engine produced a document."""
    pc = prepare(_scanned_pdf(), PDF, "scan.pdf", ocr=_MockPagedOCR(), max_chars=100000)
    assert pc.provider == "mistral_ocr"
    assert [n for n, _ in split_pages(pc.markdown)] == [1, 2]


class _MockPagedOCR:
    """Mirrors MistralOCRProvider: per-page markdown, delimited."""

    name = "mistral"

    def available(self) -> bool:
        return True

    def ocr(self, content: bytes, mime: str):
        pages = ["FACTURE page un", "FACTURE page deux"]
        md = "\n\n".join(
            f"{page_delimiter(i)}\n{t}" for i, t in enumerate(pages, start=1)
        )
        return md, [SourceRef(page=1), SourceRef(page=2)]


def test_image_branch_reports_page_count():
    """_prepare_image was the one paged branch reporting no page_count."""
    src = fitz.open()
    src.new_page().insert_text((72, 72), "ticket", fontsize=14)
    png = src[0].get_pixmap(dpi=72).tobytes("png")
    pc = prepare(png, "image/png", "shot.png", ocr=_MockOCR())
    assert pc.meta.get("page_count") == 1


# ── truncation telemetry + boundary safety ──────────────────────────────────
def test_total_chars_recorded_even_when_not_truncated():
    pc = prepare(_multipage_pdf(2), PDF, "multi.pdf", max_chars=100000)
    assert pc.truncated is False
    assert pc.meta["total_chars"] == len(pc.markdown)


def test_truncation_cuts_at_a_page_boundary_and_says_where():
    full = prepare(_multipage_pdf(5), PDF, "multi.pdf", max_chars=100000)
    half = len(full.markdown) // 2

    pc = prepare(_multipage_pdf(5), PDF, "multi.pdf", max_chars=half)

    assert pc.truncated is True
    assert pc.meta["total_chars"] == len(full.markdown)
    kept = split_pages(pc.markdown)
    assert kept, "at least one whole page must survive"
    assert pc.meta["truncated_after_page"] == kept[-1][0]
    # Every surviving delimiter is intact — a severed one breaks the consumer.
    assert pc.markdown.count("<!-- page:") == len(kept)


def test_truncation_never_leaves_a_severed_delimiter():
    """When even page 1 overruns, the cut lands mid-page — but must not bisect
    a marker."""
    pc = prepare(_multipage_pdf(3), PDF, "multi.pdf", max_chars=20)
    assert pc.truncated is True
    opened = pc.markdown.count("<!-- page:")
    complete = len(PAGE_DELIMITER_RE.findall(pc.markdown))
    assert opened == complete


def test_split_pages_degrades_on_undelimited_content():
    """0.1.x-prepared content, and non-paged sources, must not error."""
    assert split_pages("plain markdown, no markers") == [(1, "plain markdown, no markers")]
    assert split_pages("") == []

from app.ingestion.pdf_loader import PdfLoader
from app.schemas.extraction import OCRResult, TextBlock


class FakeOCR:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def recognize(self, image, *, page_number, languages):
        self.calls += 1
        return OCRResult(
            (
                TextBlock(
                    "ocr",
                    page_number,
                    "paragraph",
                    "Enzymes lower activation energy in metabolic reactions.",
                    method="ocr",
                    confidence=0.95,
                ),
            ),
            0.95,
        )


class FakeRenderer:
    def render_page(self, content, page_number, dpi):
        return b"test image"


def test_text_page_does_not_call_ocr():
    ocr = FakeOCR()
    loader = PdfLoader(ocr=ocr, renderer=FakeRenderer())
    page = loader.extract_page(b"pdf", 1, "Enzymes lower activation energy in metabolic reactions.", 0.0, True)
    assert page.status == "usable"
    assert page.classification == "text"
    assert ocr.calls == 0


def test_scan_uses_ocr_but_blank_does_not():
    ocr = FakeOCR()
    loader = PdfLoader(ocr=ocr, renderer=FakeRenderer())
    page = loader.extract_page(b"pdf", 1, "", 0.9, True)
    blank = loader.extract_page(b"pdf", 2, "", 0.0, False)
    assert page.blocks[0].method == "ocr"
    assert page.blocks[0].confidence == 0.95
    assert blank.status == "blank"
    assert ocr.calls == 1


def test_mixed_page_with_good_native_text_still_reads_large_image():
    ocr = FakeOCR()
    page = PdfLoader(ocr=ocr, renderer=FakeRenderer()).extract_page(
        b"pdf", 1, "A detailed introduction with enough native text to pass the quality gate.", 0.7, True
    )
    assert page.classification == "mixed"
    assert ocr.calls == 1
    assert "Enzymes" in " ".join(b.text for b in page.blocks)


def test_scan_without_provider_is_not_successful():
    page = PdfLoader(renderer=FakeRenderer()).extract_page(b"pdf", 1, "", 1.0, True)
    assert page.status == "failed"
    assert "ocr_unavailable" in page.warnings


def test_normalization_preserves_medical_tokens_and_lists():
    from app.ingestion.normalizer import normalize_block

    original = "Không dùng. Na+ 135.5 mmol/L; beta-blocker; IL-6; 5 mg."
    assert normalize_block(original, kind="paragraph") == original
    assert normalize_block("- First\n- Second", kind="list") == "- First\n- Second"


def test_reading_order_finishes_left_column_first():
    from app.ingestion.layout import order_blocks

    blocks = [
        TextBlock("r1", 1, "paragraph", "Right", (0.6, 0.1, 0.9, 0.2)),
        TextBlock("l2", 1, "paragraph", "Left 2", (0.1, 0.4, 0.4, 0.5)),
        TextBlock("l1", 1, "paragraph", "Left 1", (0.1, 0.1, 0.4, 0.2)),
    ]
    assert [b.block_id for b in order_blocks(blocks)] == ["l1", "l2", "r1"]


def test_pdfium_renders_and_reads_a_real_pdf():
    import io

    from reportlab.pdfgen import canvas

    from app.ingestion.pdf_renderer import PDFRenderer

    data = io.BytesIO()
    pdf = canvas.Canvas(data)
    pdf.drawString(50, 700, "Enzyme evidence")
    pdf.save()
    renderer = PDFRenderer()
    assert renderer.render_page(data.getvalue(), 1, 100).startswith(b"\x89PNG")
    blocks, image_coverage, visual = renderer.inspect_page(data.getvalue(), 1)
    assert image_coverage == 0
    assert visual
    assert "Enzyme" in " ".join(b.text for b in blocks)

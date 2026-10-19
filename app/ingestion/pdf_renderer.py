import io
import math
from contextlib import closing

from app.schemas.extraction import TextBlock


class PDFRenderer:
    def __init__(self, max_pixels: int = 25_000_000):
        self.max_pixels = max_pixels

    def render_page(self, content: bytes, page_number: int, dpi: int) -> bytes:
        import pypdfium2 as pdfium

        with pdfium.PdfDocument(content) as pdf, closing(pdf[page_number - 1]) as page:
            width, height = page.get_size()
            scale = min(dpi / 72, math.sqrt(self.max_pixels / (width * height)))
            with closing(page.render(scale=scale)) as bitmap:
                image = bitmap.to_pil()
                try:
                    output = io.BytesIO()
                    image.save(output, format="PNG")
                    return output.getvalue()
                finally:
                    image.close()

    def inspect_page(self, content: bytes, page_number: int):
        import pypdfium2 as pdfium
        from pypdfium2 import raw

        with pdfium.PdfDocument(content) as pdf, closing(pdf[page_number - 1]) as page:
            width, height = page.get_size()
            area, visual = 0.0, False
            for obj in page.get_objects():
                visual = True
                if obj.type == raw.FPDF_PAGEOBJ_IMAGE:
                    left, bottom, right, top = obj.get_bounds()
                    area += max(0, right - left) * max(0, top - bottom)
            blocks = []
            with closing(page.get_textpage()) as text_page:
                for index in range(text_page.count_rects()):
                    left, bottom, right, top = text_page.get_rect(index)
                    text = text_page.get_text_bounded(left, bottom, right, top).strip()
                    if text:
                        blocks.append(
                            TextBlock(
                                f"p{page_number}-native{index}",
                                page_number,
                                "paragraph",
                                text,
                                (
                                    max(0, left / width),
                                    max(0, 1 - top / height),
                                    max(0, right / width),
                                    max(0, 1 - bottom / height),
                                ),
                            )
                        )
            return blocks, min(1, area / (width * height)), visual

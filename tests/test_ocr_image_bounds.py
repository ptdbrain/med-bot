import io

import pytest


def test_ocr_input_is_resized_to_configured_pixel_budget():
    image_module = pytest.importorskip("PIL.Image")
    from app.ingestion.ocr.paddle import limit_image_pixels

    source = image_module.new("RGB", (2400, 2400), "white")
    encoded = io.BytesIO()
    source.save(encoded, format="PNG")
    bounded = limit_image_pixels(encoded.getvalue(), max_pixels=1_000_000)
    with image_module.open(io.BytesIO(bounded)) as result:
        assert result.width * result.height <= 1_000_000

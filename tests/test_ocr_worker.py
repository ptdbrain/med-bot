import io
import json

from app.ingestion.ocr.worker import serve


class FakeProvider:
    def recognize(self, image, *, page_number, languages):
        return {"page": page_number, "image": image.decode(), "languages": list(languages)}


def test_worker_returns_one_serialized_page_result():
    request = {"image": "YQ==", "page_number": 3, "languages": ["vi", "en"]}
    source = io.StringIO(json.dumps(request))
    output = io.StringIO()
    providers = []

    def factory():
        provider = FakeProvider()
        providers.append(provider)
        return provider

    serve(source, output, factory)
    assert len(providers) == 1
    assert json.loads(output.getvalue()) == {"page": 3, "image": "a", "languages": ["vi", "en"]}

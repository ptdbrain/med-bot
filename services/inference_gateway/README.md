# Inference gateway

Run from the repository root with local provider settings and a nonempty
HTTP_PROVIDER_KEY. Never point this gateway back to itself through HTTP providers.

```powershell
python -m uvicorn services.inference_gateway.app:app --env-file .env --host 127.0.0.1 --port 8010
```

The versioned endpoints are `/v1/ocr`, `/v1/embeddings`, `/v1/rerank`,
`/v1/generate`, `/v1/verify`. They require Bearer authentication. The authenticated
`/v1/health` reports the embedding fingerprint that the client must configure.
Use TLS termination and request/concurrency limits before remote exposure.
This gateway has not been deployed to an external cloud account by this task.

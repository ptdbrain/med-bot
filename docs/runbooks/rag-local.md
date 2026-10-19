# Local RAG Runbook

This runbook separates the deterministic demo profile from the local quality profile. Neither profile is a medical diagnostic or prescribing service.

## Prerequisites

- Python 3.11; Node.js 20+ for the Next.js UI.
- For `local`: Ollama, Qdrant, sentence-transformers, PaddleOCR, PaddlePaddle, PDFium, and the configured model weights. First-time model downloads need network access and disk space.
- Do not put patient records or secrets in this repository. Use synthetic or authorized educational documents only.

## Install And Check

```powershell
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,rag]"
uv pip install -r requirements-ocr.txt
Copy-Item .env.example .env
```

Set `RAG_PROFILE=demo` for deterministic offline development. For quality mode, set `RAG_PROFILE=local`; start Ollama and Qdrant first, then explicitly pull the model configured by `OLLAMA_MODEL`. `python -m scripts.doctor --profile demo --json` checks configuration without downloading models. `python -m scripts.doctor --profile local --probe-models --json` checks local services and model availability; missing services are blockers, not a fallback to mocks.

## Start Services

```powershell
docker compose up -d qdrant
ollama serve
```

In separate terminals:

```powershell
$env:RAG_PROFILE="local"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd web
npm ci
npm run dev
```

Open `http://localhost:3000`. API liveness is `http://localhost:8000/health`; readiness is `/ready`. Ingestion jobs are durable in the metadata database; run the worker in another terminal for queued jobs:

```powershell
$env:RAG_PROFILE="local"
python -m scripts.ingest_worker
```

## Ingest And Query

Upload through the Ingest page or submit a PDF/document through the API. Keep the returned `doc_id` to replace or delete that exact source. Same filenames are not replacement identities. Scan PDFs require OCR; check extraction page status and warnings before trusting search results.

```powershell
python -m scripts.check_index --read-only --json
python -m scripts.validate_eval_dataset --corpus evaluation/corpus/manifest.json --cases evaluation/cases
```

After changing embedding model/revision, chunking policy, or OCR processing, build a new compatible index or explicitly re-ingest; do not mix vectors with different embedding fingerprints. Publication keeps the previous active revision visible until the new one is committed. For a failed job, inspect its report and retry only after fixing the reported stage. Delete is document-scoped and removes its active chunks.

## Evaluation

Evaluation distinguishes **development regression checks** (pipeline verification using draft data) from **release candidate quality evaluations** (clinical ground-truth validation).

### 1. Development Regression (Pipeline Verification)

All 120 current synthetic evaluation cases in `evaluation/cases/` are assistant-authored drafts (`reviewed_by: null`). To verify pipeline mechanics, retrieval algorithms (BM25, Dense, Hybrid, Reranking), and ranking metrics calculation without halting at dataset review validation, run with `--allow-draft`:

```powershell
python -m scripts.run_benchmark --config config/benchmarks/rag-quality.json --split dev --output data/reports/rag-dev.json --allow-draft
```

For rapid local testing during development, add `--limit <N>`:

```powershell
python -m scripts.run_benchmark --config config/benchmarks/rag-quality.json --split dev --output data/reports/rag-dev.json --allow-draft --limit 10
```

> [!WARNING]
> **Quality Gate Status: `not_evaluated_unreviewed`**
> Reports generated with `--allow-draft` explicitly retain `"dataset_reviewed": false` and `"quality_gate": "not_evaluated_unreviewed"`. Successful execution with exit code 0 verifies pipeline execution mechanics only; **it is not an accuracy claim and must never be presented as a clinical quality certification or production sign-off.**

### 2. Release Evaluation Gate (Clinical Ground Truth)

Official quality gate certification for release candidates strictly disallows draft cases. Running without `--allow-draft` enforces that every evaluation case has been reviewed and signed off by a clinical domain expert (`reviewed_by != null`):

```powershell
# Pre-flight check: verify that all cases in dataset are reviewed
python -m scripts.validate_eval_dataset --corpus evaluation/corpus/manifest.json --cases evaluation/cases --require-review

# Official release evaluation on frozen held-out test split (requires reviewed cases)
python -m scripts.run_benchmark --config config/benchmarks/rag-quality.json --split test --output data/reports/rag-release.json
```

- If any case in the dataset is unreviewed, this command intentionally halts with `ValueError: Invalid evaluation dataset: ...:unreviewed` (exit code 1) to block unauthorized release.
- Never tune hyper-parameters on the held-out test split. Test evaluation must be run exactly once against a frozen release candidate.
- Real-world clinical quality acceptance requires verified human ground-truth labels, a live local model stack (`RAG_PROFILE=local`), and human claim/citation audit.

## Troubleshooting

- `/ready` is 503: read the per-stage checks; start the named service or install the optional provider package. A missing embedding model is not downloaded by doctor.
- OCR worker error: verify OCR dependencies and local model cache; ingest reports should retain page warnings. OCR confidence is not recognition accuracy.
- Ingested document but zero chunks: treat as failed extraction, inspect the job/page report, and do not assume it was indexed.
- Irrelevant answer: inspect `/retrieve` candidates and signals, relevance decision reasons, and source pages. Do not lower a threshold solely to force an answer.
- Incompatible vector index: use a new index/collection generation and re-ingest. Do not delete an existing index without a verified backup and explicit operator intent.

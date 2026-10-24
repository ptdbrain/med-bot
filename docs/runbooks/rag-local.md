# Local RAG Quality Runbook

## Profiles

`config/profiles/demo.env` is deliberately mock/lexical/in-memory. Use it for
the UI demo and offline smoke tests only. `config/profiles/local-quality.env.example`
selects Ollama, Sentence Transformers BGE-M3, Qdrant, and the multilingual
cross-encoder. Load the chosen profile into the current PowerShell session
before starting the API or running CLI commands; this does not replace `.env`:

```powershell
Get-Content config/profiles/local-quality.env.example | Where-Object { $_ -match '^\s*[^#].*=.*$' } | ForEach-Object {
  $name, $value = $_ -split '=', 2
  Set-Item "Env:$name" $value
}
```

For the demo, use `config/profiles/demo.env` in the same command. Do not commit
credentials.

The API and normal CLI commands, including doctor, read `RAG_PROFILE` (default
`demo`). Doctor's explicit `--profile` takes precedence. Provider precedence is profile defaults, process environment,
then supported `--set KEY=VALUE` overrides; unknown override keys are rejected.
Neither command reads or rewrites `.env` implicitly. Loading a profile does not
clear other environment variables: use a new PowerShell session for independent
runs. The local example uses `data/processed/local-quality` and a separate Qdrant
collection to avoid mixing with the existing demo index.

## Runtime Doctor

```powershell
python -m scripts.doctor --profile demo
python -m scripts.doctor --profile local --json --output data/processed/runtime-smoke/runtime.json
python -m scripts.doctor --profile local --probe-models --json
```

Exit code `1` means missing/error; `0` means no reported configuration blockers,
not a release-quality certification. Without `--probe-models`, dependency/cache
checks do not prove inference. With the flag, cached models are loaded using
`local_files_only=True`; no model weights are downloaded. Ollama inference is a
small generation request. An absent Qdrant collection is reported separately
from an unreachable service; doctor never creates collections.

`GET /health` remains liveness. `GET /ready` runs lightweight configured-provider
checks and returns `503` on blockers. It does not run costly model inference or
certify index consistency; use the explicit probe and index check for those.
The current application still eagerly constructs its vector store at startup,
so an unavailable Qdrant may prevent startup before `/health` can be served.

The six cases in `app/evaluation/eval_dataset.json` are a draft smoke set, not
reviewed labels and not a release benchmark. Run:

```powershell
python -m scripts.run_eval --split dev
python -m scripts.check_index --json
```

The eval command prints `DRAFT` and the active configured providers. It does
not claim dense retrieval or cross-encoder results unless those providers are
selected and successfully instantiated by the application. The current eval
is not an ablation comparison between BM25, dense, hybrid, and reranked modes.

## Start Services

Start Docker Desktop first, then:

```powershell
docker compose up -d qdrant
Invoke-RestMethod http://localhost:6333/healthz
```

Start Ollama separately and verify it is responding:

```powershell
ollama list
Invoke-RestMethod http://localhost:11434/api/tags
```

The model names in the local profile are configuration targets. A successful
service health check is not proof of inference; run embedding, reranker, and
generation probes after models are available. Install the optional embedding
and reranker dependency before probing:

```powershell
python -m pip install -r requirements-rag.txt
```

The repository's current PDF
loader only extracts a PDF text layer; scanned-PDF OCR and OCR CER/WER
benchmarking are not implemented, so scanned PDF ingestion is not release
ready.

## Ingest Smoke Document

The repeatable CPU-only smoke below uses a fresh SQLite dense index, **not**
Qdrant or Ollama. It does not need an API server or call any LLM. A BGE-M3 model
snapshot must already be cached. It ingests two synthetic Markdown documents,
checks persisted IDs and finite nonzero float32 vectors, opens new store objects,
then requires both Vietnamese and English queries to return the expected quote.
This is a store-reopen check, not a process-crash/recovery or held-out benchmark.

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONUTF8='1'
.\.venv-rag-quality\Scripts\python.exe -m scripts.smoke_dense_ingest --json
```

Each default run creates `data/processed/runtime-smoke/<unique-id>/report.json`.
`--run-dir PATH` must name a new directory; existing directories are never
overwritten. Failure returns nonzero and records the failing stage. Reports
include source hashes, model snapshot, package versions, timings, vector checks,
and query/evidence. Generation, reranking, OCR, and active-revision gaps remain
explicit, even when the dense smoke passes.

On this Windows machine `.venv-rag-quality` was created with
`python -m venv --system-site-packages .venv-rag-quality`, then
`sentence-transformers==6.2.0` was installed there, reusing existing Torch
`2.8.0+cpu`. The live run used Transformers `5.20.2` and Python `3.13.7`.
This environment inherits system packages; it is not a clean-machine lockfile.
Do not reinstall Torch just to repeat this smoke. The optional dependency file's
Torch pin has not been validated by this run.

Use an isolated local data directory and collection for the quality run. Do not
clear or overwrite an existing index. After configuring the local profile and
starting the API, ingest one small Markdown file through `POST /ingest/file`,
then verify `GET /documents`, search/retrieve the expected phrase, and run
`python -m scripts.check_index --json`. Record the collection, model IDs,
corpus revision, and exact query/evidence in the report.

The current index schema has no active-revision publication pointer. Therefore
the index check validates non-empty counts and metadata/vector count agreement,
but cannot certify active-revision atomicity or crash recovery.

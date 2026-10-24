# Runtime And Dense Ingestion Evidence

Date: 2026-10-24 (Asia/Saigon). Working branch: `codex/runtime-smoke`.
Source baseline: `43f1371` plus uncommitted runtime/smoke changes.
This is not a release-quality PASS.

## Live Evidence

Artifacts are local generated files, excluded by `data/processed/*`:

- `data/processed/runtime-smoke/2026-10-24-runtime.json`: doctor, `ready=false`, exit 1.
- `data/processed/runtime-smoke/2026-10-24-bge-m3/report.json`: first dense ingest run, PASS.
- `data/processed/runtime-smoke/2026-10-24-bge-m3-offline/report.json`: repeated run with loader-enforced `local_files_only=True`, PASS.

Both dense runs use real CPU BGE-M3 embeddings, not lexical or injected test
embeddings. Hugging Face/Transformers offline environment flags were also set.

| Check | Result | Evidence |
| --- | --- | --- |
| Model | PASS | `BAAI/bge-m3`, snapshot `5617a9f61b028005a4858fdac845db406aefb181` |
| Corpus | PASS | Two synthetic Markdown files: storage conventions and unrelated weather observations |
| Ingest | PASS | 2 documents, 2 chunks, 2 persisted float32 vectors, dimension 1024 |
| Index IDs | PASS | No missing/orphan chunk IDs between SQLite metadata and dense store |
| Vector values | PASS | Finite, nonzero, norms approximately 1.00000004 |
| Vietnamese query | PASS | Top result contains `SMOKE-2026`, cosine 0.748310 on both runs |
| English query | PASS | Top result contains the same Vietnamese evidence, cosine 0.701421 on both runs |
| Reopen | PASS | New metadata/vector-store objects retrieve expected evidence without re-ingesting documents |
| Ollama | NOT READY | No response from the configured local service; generator model/inference not verified |
| Qdrant | NOT READY | No response from configured service; prior Docker service start was denied |
| Cross-encoder | MISSING | Snapshot directory has config only, no usable model weights; inference not run |
| PDF OCR | NOT IMPLEMENTED | Current loader has text-layer extraction only |
| Active revision | NOT IMPLEMENTED | No active-publication pointer or crash-safe revision evidence |
| Answer/safety quality | NOT EVALUATED HERE | This smoke never calls an LLM and has no reviewed medical-answer labels |

Evidence quote: "Mã kiểm thử lưu trữ là SMOKE-2026. Tài liệu được giữ trong 30 ngày."
The unrelated weather document is a distractor. Two queries on two synthetic
documents demonstrate plumbing only, not general retrieval accuracy.

## Timing And Environment

Repeated offline run: ingestion including model initialization 27.074 seconds;
Vietnamese query 0.141 seconds; English query 0.151 seconds; total 27.502 seconds.
First run ingestion took 148.981 seconds. These are individual measurements,
not p50/p95 or a controlled cold/warm benchmark.

Windows 11, Python 3.13.7, Torch 2.8.0 (CPU), Sentence Transformers 6.2.0,
Transformers 5.20.2. Dependencies were installed in `.venv-rag-quality` with
system-site-packages; the system Torch installation was reused. This is not a
fully isolated, reproducible clean-machine dependency lock.

Source SHA256, exact model path, retrieved chunk IDs, scores, timings and package
versions are recorded in each report. Existing `data/indexes` was not reindexed
or cleared by this smoke. Reopening store objects is not a process restart,
crash injection, rollback, or atomic publication test.

## Next Gates

Make Docker/Qdrant and Ollama available, install the configured generator and
reranker weights explicitly, then run doctor with `--probe-models` and repeat
ingestion in an isolated Qdrant collection. Implement active revisions and OCR
before claiming the original P0 ingestion/release acceptance is complete.
Reviewed retrieval/claim-level answer labels and release-wide checks remain
separate work. See [the runbook](../runbooks/rag-local.md) for commands.

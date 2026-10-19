# Implementation ledger: 2026-09-22-rag-quality-completion

Plan: ../superpowers/plans/2026-09-22-rag-quality-completion.md
Baseline: 43f1371; isolated checkout codex/rag-quality.
Baseline tests: 94 passed, 1 skipped, 4 subtests (Python 3.13).
Latest verification after upstream integration: backend 191 passed, 1 skipped, 4 subtests; reviewed dataset validator 16/16; Ruff, OCR-scope Black, and Docker Compose config passed. Repo-wide Black check still reports six upstream formatting issues.

## Decisions

- Use workspace-local pytest basetemp because the global Windows pytest temp directory is inaccessible.
- Preserve existing indexes and source documents in the original checkout. Model downloads and paid APIs are not required by offline tests.
- No commits or push requested; implementation changes remain reviewable in the worktree.
- Dataset labels authored by the assistant must not be presented as human-reviewed medical ground truth.
- Pre-flight: extraction types feed normalization/chunking; parent text must retain original spans rather than concatenate overlapping chunks.
- Pre-flight: publication must bind vector, metadata and BM25 revisions; cloud/local embeddings cannot share incompatible indexes.
- Pre-flight: generation and verifier share one repair budget; citation validity is distinct from entailment.

## Tasks

1. Runtime profiles/readiness: implemented; demo ready. Local is blocked by missing Ollama and Qdrant; provider model availability is not fully probed.
2. Evaluation corpus/labels: 120 synthetic cases and fixtures built; 0 reviewed. Human-quality gate remains open.
3. Extraction contracts/identity: implemented and covered by tests.
4. PDF/OCR: text/scan/mixed routing and supervised per-PDF JSONL worker implemented; request batches are capped at 64, only timeout gets one retry, and crash diagnostics are correlated to jobs/pages. A two-page English scan stayed in one worker (requests 1 and 2); Vietnamese fixture still fails intermittently, and 250 DPI caused native exits. CER/WER are not established; OCR gate FAIL.
5. Normalization/layout: core normalization, repeated-margin filtering, and column ordering implemented; comprehensive table/layout gold validation remains open.
6. Token chunking: token budget chunker implemented and unit tested; live embedding tokenizer/truncation verification remains open.
7. Index publication: staged revisions, active revision filtering, and rollback tests implemented; process-kill/Qdrant failure-injection matrix remains open.
8. Ingest jobs/UI: durable queue, retry/cancel API, worker, and UI polling implemented; browser end-to-end/crash-resume workflow remains open.
9. Hybrid/query processing: BM25, dense, RRF, query normalization/transform, and retrieval tests implemented; full same-corpus benchmark comparison remains open.
10. Reranker/relevance: lexical and cross-encoder adapters, score signals, and relevance policy implemented; live cross-encoder calibration remains open.
11. Context packing: parent-child expansion and conservative context budgeting implemented; production tokenizer live validation remains open.
12. Answer/citation validation: structured generation validation, shared repair, citation-ID checks, safety routing, and tests implemented; live Ollama answer verification remains open.
13. Claim support: verifier interface and strict-mode handling implemented; required reviewed claim-pair audit is not available.
14. Cloud adapters: versioned HTTP adapters and gateway implemented/tested with fakes; live cloud endpoint is not configured.
15. Benchmark: dataset validation and metrics/benchmark scaffolding implemented. Draft results are not quality acceptance; human audit, full answer metrics, tuning, and held-out release run remain open.
16. Release verification: runbook, release checklist, quality report, smoke contract, read-only index inspection, and Compose validation added. Full live E2E, scale, and hardware performance gates remain open.

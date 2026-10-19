# RAG Quality Report

## Status

This is an implementation-progress snapshot, not a clinical accuracy claim. The default `evaluation/cases/` set remains assistant-authored synthetic drafts with no human reviewer. A separate `evaluation/reviewed_education/` dataset now records clinician reviewers and passes structural validation, but no benchmark run has yet established answer faithfulness or clinical accuracy. Those release gates remain `NOT_EVALUATED`.

## Verified In This Workspace

- Backend suite: 191 passed, 1 skipped, 1 warning, 4 subtests (Python 3.11, local `.venv`, 2026-10-19; includes the clinician-reviewed dataset tests).
- Web suite: 12 passed across 6 files; ESLint and production build passed.
- Static checks: Ruff, OCR-scope Black check, Docker Compose config, and `git diff --check` passed. Repo-wide Black check currently reports six formatting issues in files from the intervening upstream evaluation commits; the OCR change does not touch those files.
- Demo doctor passed; local doctor correctly reported Ollama and Qdrant unavailable. Read-only index check found an empty metadata database (0 documents, 0 chunks).
- OCR live fixture checks: English scan succeeded at 160 DPI; Vietnamese scan returned no usable text at 160 DPI. Both scans failed at 250 DPI due worker native process exits. A previous 160-DPI Vietnamese extraction had plausible confidence (~0.987) but lost diacritics; these inconsistent runs show the OCR path is not reliable enough. The pipeline now includes page warning codes in the failure message. OCR quality gate remains FAIL; character/word error rates were not established.
- The default was conservatively set to 160 DPI and a 2-million-pixel cap after 250-DPI native crashes; this is a runtime stability adjustment, not evidence that lower-resolution OCR meets the required recognition quality.
- After adding the supervised per-PDF worker, a duplicated two-page English scan completed in one process on three runs. Latest run: 72.5s; `worker_request` values 1 and 2 shared one correlation ID, timeout 180s, and batch limit 64. Earlier run measured 41.1s and 33.6s inference per page. Resize was 1323x1872 to 1188x1682; PaddleOCR 3.7.0 / PaddlePaddle 3.3.1. This validates the multi-page process lifecycle only, not OCR accuracy or Vietnamese quality.
- Synthetic benchmark draft: labels remain unreviewed, so those results are not an estimate of real-world accuracy. The separate reviewed set passes `validate_eval_dataset --require-review` (16 cases), but retrieval/answer benchmarks on it are still NOT_RUN.
- Live LLM, Qdrant, full UI workflow, hosted gateway, scale, and held-out audited test: NOT_RUN in this environment.

## Interpretation

Retrieval and ingestion code paths have automated unit coverage, but this is not evidence that the system answers medical questions accurately. Do not use OCR confidence as CER/WER. Do not present assistant-drafted labels or mock generation as human-validated correctness. The required next quality inputs are reviewed Vietnamese/English scan references, reviewed support/citation labels, and a live configured local generation stack.

## Artifacts

- Dataset and corpus: `evaluation/` and `data/quality-benchmark/`.
- Draft benchmark output: `data/reports/` (ignored runtime artifacts).
- Current gates: `docs/quality/release-checklist.md`.
- Operator procedure: `docs/runbooks/rag-local.md`.

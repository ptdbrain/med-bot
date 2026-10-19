# RAG Release Checklist

Report each item as PASS, FAIL, or NOT_RUN and attach current artifacts. A mock provider or synthetic corpus cannot satisfy a live-model or human-ground-truth gate.

- [ ] Runtime doctor passes for the declared profile; health and readiness differ correctly.
- [ ] Clean-text, scanned, and mixed PDFs ingest with page-level provenance; OCR Vietnamese CER/WER and numeral/negation errors are measured.
- [ ] Token chunking stays within the configured model tokenizer budget; parent text and source spans are intact.
- [ ] Failure injection proves old active revisions remain queryable through embed/upsert/metadata errors and restart.
- [ ] BM25, dense, hybrid, hybrid+reranker are compared on the same reviewed cases; Recall@5/10, MRR, nDCG, latency and failure denominators are included.
- [ ] Answerable coverage, false refusal, unanswerable refusal, citation validity, human citation correctness, claim support, disclaimer and safety are separately reported.
- [ ] Human-reviewed train/dev/test labels are split by evidence group; held-out test is run once for a frozen release candidate.
- [ ] Local LLM, embedding, OCR and reranker live smoke passes on declared hardware. Cloud remains NOT_RUN until an authorized endpoint is configured.
- [ ] Full backend and web tests, lint, format, Docker Compose config, smoke and read-only index checks pass.
- [ ] Backup/restore and operator runbook are checked; no secrets or sensitive records enter artifacts.

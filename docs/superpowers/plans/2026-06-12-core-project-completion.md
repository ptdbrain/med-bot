# Core Project Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the basic end-to-end MedTutor feature set described in `mota.md`, replacing visible placeholders with tested backend and frontend workflows.

**Architecture:** Keep ingestion, retrieval, RAG orchestration, safety, evaluation, jobs, API, and UI as separate modules. Advanced RAG modes compose the existing vector store, BM25 store, reranker, query transformer, metadata store, LLM provider, and medical guardrails rather than duplicating retrieval or generation logic.

**Tech Stack:** Python 3.11, FastAPI, SQLite, Qdrant, BGE-M3, Ollama/Qwen3, Next.js 16, React 19, TypeScript, unittest/pytest-compatible tests, Vitest.

---

### Task 1: Complete advanced RAG modes

**Files:**
- Modify: `app/rag/rerank_rag.py`
- Modify: `app/rag/parent_child_rag.py`
- Modify: `app/rag/multi_query_rag.py`
- Modify: `app/rag/corrective_rag.py`
- Modify: `app/rag/agentic_router.py`
- Modify: `app/retrieval/query_transform.py`
- Modify: `app/rag/service.py`
- Test: `tests/test_advanced_rag.py`

- [ ] Add failing tests proving rerank retrieves a larger candidate set, parent-child expands a matching section, multi-query fuses multiple result lists, corrective retries a weak query, and agentic routing delegates safely.
- [ ] Run `python -m unittest tests.test_advanced_rag -v` and confirm failures are caused by the unimplemented modes.
- [ ] Implement each mode through injected retrieval, reranking, query transformation, generation, and guardrail collaborators.
- [ ] Register `rerank`, `parent_child`, `multi_query`, `corrective`, and `agentic` in `RAGService`.
- [ ] Re-run targeted and existing RAG tests.

### Task 2: Persist parent sections and expose document management

**Files:**
- Modify: `app/schemas/document.py`
- Modify: `app/ingestion/chunker.py`
- Modify: `app/retrieval/metadata_store.py`
- Modify: `app/ingestion/pipeline.py`
- Modify: `app/api/ingest.py`
- Create: `app/api/documents.py`
- Modify: `app/main.py`
- Test: `tests/test_document_management.py`

- [ ] Add failing tests for deterministic parent IDs, persisted parent sections, document listing, URL ingest synchronization, and document deletion.
- [ ] Run the targeted tests and confirm the missing behavior fails.
- [ ] Add parent-section persistence and metadata-store document queries without coupling API routes to SQLite.
- [ ] Add `GET /documents` and `DELETE /documents/{doc_id}` service-backed endpoints.
- [ ] Re-run ingestion, retrieval, and API tests.

### Task 3: Complete multimodal ingest boundaries

**Files:**
- Modify: `app/ingestion/image_processor.py`
- Modify: `app/ingestion/video_processor.py`
- Modify: `app/rag/service.py`
- Create: `app/api/video.py`
- Modify: `app/main.py`
- Test: `tests/test_multimodal_processing.py`

- [ ] Add failing tests for OCR/caption provider injection, image metadata, validated video upload, transcript ingestion, and unsupported files.
- [ ] Implement provider protocols with deterministic local/mock defaults.
- [ ] Add `POST /video/ingest` and service methods.
- [ ] Verify image, audio, and video content becomes searchable.

### Task 4: Complete knowledge-update jobs

**Files:**
- Modify: `app/jobs/crawler.py`
- Modify: `app/jobs/incremental_update.py`
- Modify: `app/jobs/scheduler.py`
- Modify: `app/rag/service.py`
- Modify: `app/api/ingest.py`
- Test: `tests/test_knowledge_update.py`

- [ ] Add failing tests for safe public HTTP URLs, content hashing, unchanged-content skipping, changed-content re-ingest, and manual batch execution.
- [ ] Implement a small crawler adapter and SQLite-backed URL state.
- [ ] Route `/ingest/url` through the service so BM25 and vector indexes stay synchronized.
- [ ] Verify repeated unchanged URLs do not create duplicate work.

### Task 5: Complete evaluation output

**Files:**
- Modify: `app/evaluation/retrieval_eval.py`
- Modify: `app/evaluation/answer_eval.py`
- Modify: `app/evaluation/safety_eval.py`
- Modify: `app/api/eval.py`
- Modify: `app/evaluation/eval_dataset.json`
- Test: `tests/test_evaluation.py`

- [ ] Add failing tests for per-case details, selected RAG mode, hit@k, MRR, citation, insufficient-context, disclaimer, emergency, diagnosis, and prescription checks.
- [ ] Return aggregate metrics plus auditable per-question details.
- [ ] Expand the dataset with comparison and multimodal educational cases.
- [ ] Run evaluation against ingested sample data.

### Task 6: Replace disabled Next.js pages with working workflows

**Files:**
- Create: `web/lib/api/ingest.ts`
- Create: `web/lib/api/documents.ts`
- Create: `web/lib/api/evaluation.ts`
- Modify: `web/app/ingest/page.tsx`
- Modify: `web/app/documents/page.tsx`
- Modify: `web/app/evaluation/page.tsx`
- Modify: `web/features/chat/chat-workspace.tsx`
- Modify: `web/stores/workspace-store.ts`
- Test: `web/lib/api/ingest.test.ts`
- Test: `web/lib/api/documents.test.ts`
- Test: `web/lib/api/evaluation.test.ts`

- [ ] Add failing API-client and component-state tests.
- [ ] Implement document, URL, image, audio, and video ingest controls with progress and errors.
- [ ] Implement searchable document listing and deletion.
- [ ] Implement evaluation execution and metric/detail rendering.
- [ ] Expose all supported RAG modes in chat.
- [ ] Run Vitest, ESLint, and a production Next.js build.

### Task 7: Documentation and completion audit

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `frontend/streamlit_app.py`

- [ ] Update the README implemented-mode list, API table, setup, evaluation, demo, limitations, and roadmap.
- [ ] Keep Streamlit compatible with every supported mode and ingest endpoint.
- [ ] Run all backend tests, frontend tests, lint, compile checks, Docker config validation, and production builds.
- [ ] Start the stack and smoke-test health, ingest, each RAG mode, emergency routing, document management, evaluation, and the Next.js pages.
- [ ] Compare every `mota.md` requirement with direct evidence and leave the project goal active if any required item remains incomplete.

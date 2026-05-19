# Phase 1 Naive RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Phase 1 MVP for MedTutor with text ingestion, chunking, local retrieval, mock LLM answers, citations, safety guardrails, API, Streamlit demo, sample data, docs, and tests.

**Architecture:** FastAPI owns HTTP endpoints and delegates to `RAGService`. Ingestion loaders create `Document` objects, `TextChunker` creates metadata-rich chunks, `InMemoryVectorStore` provides offline lexical retrieval, and `NaiveRAG` coordinates guardrails, retrieval, and answer generation.

**Tech Stack:** Python 3.11, FastAPI, Streamlit, SQLite, local vector fallback, Pydantic API models, pytest, Docker.

---

### Task 1: Red Tests

**Files:**
- Create: `tests/test_chunker.py`
- Create: `tests/test_safety.py`
- Create: `tests/test_retrieval.py`
- Create: `tests/test_api.py`

- [x] Write tests for chunk metadata, emergency detection, diagnosis/prescription blocking, retrieval ranking, chat API citations, and empty-context answer.
- [x] Run `python -m pytest tests -q`.
- [x] Expected before dependencies/code: fail because pytest/app modules are unavailable.

### Task 2: Core Domain and Ingestion

**Files:**
- Create: `app/schemas/document.py`
- Create: `app/ingestion/*.py`
- Create: `app/utils/*.py`

- [x] Add domain dataclasses for documents, chunks, metadata, and search results.
- [x] Add file validation, hashing, text/html/pdf/docx loaders, and metadata-aware chunking.
- [x] Add image/audio/video processor boundaries with mock providers for later phases.

### Task 3: Retrieval and Storage

**Files:**
- Create: `app/retrieval/vector_store.py`
- Create: `app/retrieval/metadata_store.py`
- Create: `app/retrieval/bm25_store.py`
- Create: `app/retrieval/reranker.py`
- Create: `app/retrieval/query_transform.py`

- [x] Implement deterministic local retrieval with normalized token vectors.
- [x] Persist vector chunks to JSON and metadata to SQLite.
- [x] Keep Qdrant and advanced retrieval as clear adapter boundaries.

### Task 4: Safety and RAG

**Files:**
- Create: `app/safety/*.py`
- Create: `app/llm/*.py`
- Create: `app/rag/*.py`

- [x] Add emergency detection, PII redaction, medical guardrails, and required disclaimer.
- [x] Add mock LLM answer generation with source references.
- [x] Add `NaiveRAG` and `RAGService`; advanced modes return explicit Phase 1 unavailable responses.

### Task 5: API and UI

**Files:**
- Create: `app/main.py`
- Create: `app/api/*.py`
- Create: `frontend/streamlit_app.py`

- [x] Add `/health`, `/ingest/file`, `/ingest/url`, `/chat`, `/voice/transcribe`, `/voice/chat`, `/image/ingest`, and `/eval/run`.
- [x] Add Streamlit tabs for chat, document upload, image upload, audio upload, and sample-ingest instructions.

### Task 6: Docs, Samples, and Deployment

**Files:**
- Create: `data/sample/*.md`
- Create: `README.md`
- Create: `.env.example`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `AGENTS.md`
- Create: `requirements.txt`
- Create: `pyproject.toml`

- [x] Add fake/public-style medical education sample documents.
- [x] Document safety scope, architecture, usage, eval, roadmap, and portfolio highlights.
- [x] Add Docker and dependency metadata.

### Task 7: Verification

**Files:**
- Read: project files

- [ ] Run `python -m pytest tests -q` after dependencies are installed.
- [ ] Run `python -m compileall app scripts -q`.
- [ ] Run a backend smoke check with `uvicorn app.main:app --reload`.

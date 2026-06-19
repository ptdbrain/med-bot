# Real-Model RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional BGE-M3, Qdrant, and Ollama production path while preserving the current offline defaults.

**Architecture:** Provider factories assemble the LLM, embedding, and vector-store adapters from `Settings`. Dense retrieval uses BGE-M3 vectors in Qdrant, sparse retrieval keeps the existing BM25 implementation, and hybrid mode fuses both rankings before sending grounded context to Ollama.

**Tech Stack:** Python 3.11+, sentence-transformers, Qdrant HTTP API, Ollama `/api/chat`, unittest/pytest-compatible tests.

---

### Task 1: Configuration and Ollama provider

**Files:**
- Modify: `app/config.py`
- Create: `app/llm/ollama_llm.py`
- Create: `app/llm/factory.py`
- Test: `tests/test_model_providers.py`

- [ ] Add failing tests for environment parsing, provider selection, prompt
      construction, successful Ollama responses, and HTTP failures.
- [ ] Run `python3 -m unittest tests.test_model_providers -v` and verify failure
      because the provider modules do not exist.
- [ ] Implement the minimal settings fields, prompt builder, HTTP adapter, and
      factory needed by the tests.
- [ ] Re-run the targeted tests and verify they pass.

### Task 2: BGE-M3 embedding adapter

**Files:**
- Create: `app/embeddings/__init__.py`
- Create: `app/embeddings/base.py`
- Create: `app/embeddings/sentence_transformer.py`
- Create: `app/embeddings/factory.py`
- Test: `tests/test_embedding_provider.py`

- [ ] Add failing tests using an injected fake sentence-transformer model.
- [ ] Verify the tests fail because the embedding adapter is missing.
- [ ] Implement lazy loading, normalized encoding, document/query methods, and
      actionable dependency errors.
- [ ] Verify the targeted tests pass without downloading a model.

### Task 3: Qdrant vector store

**Files:**
- Modify: `app/retrieval/vector_store.py`
- Test: `tests/test_qdrant_vector_store.py`

- [ ] Add failing tests using a fake HTTP session for collection creation,
      deterministic point payloads, and search-result reconstruction.
- [ ] Verify the tests fail for the missing implementation.
- [ ] Implement the Qdrant REST adapter and a narrow vector-store protocol.
- [ ] Verify Qdrant tests and existing lexical retrieval tests pass.

### Task 4: Service integration

**Files:**
- Modify: `app/retrieval/metadata_store.py`
- Modify: `app/ingestion/pipeline.py`
- Modify: `app/rag/service.py`
- Modify: `app/api/health.py`
- Test: `tests/test_real_model_service.py`
- Modify: `tests/test_hybrid_rag.py`

- [ ] Add failing tests for provider injection, BM25 rebuild from SQLite, and
      hybrid dense/sparse fusion independent of the concrete vector store.
- [ ] Implement factories and protocol-based service assembly.
- [ ] Keep default `Settings()` behavior on mock/lexical providers.
- [ ] Run all dependency-light backend tests.

### Task 5: Runtime setup and documentation

**Files:**
- Create: `requirements-rag.txt`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `README.md`

- [ ] Add the optional sentence-transformers dependency.
- [ ] Document Ollama startup, `qwen3:4b` pull, Qdrant startup, model-mode
      environment variables, first ingest, and health checks.
- [ ] Keep Docker backend pointed at `qdrant:6333` and host development pointed
      at `localhost:6333`.

### Task 6: Verification

- [ ] Run targeted new tests.
- [ ] Run existing backend unit tests that do not require unavailable packages.
- [ ] Run `python3 -m compileall -q app tests`.
- [ ] Check Ollama and Qdrant endpoints.
- [ ] If runtime services are available, execute one ingest/query smoke test;
      otherwise report the exact remaining setup boundary.

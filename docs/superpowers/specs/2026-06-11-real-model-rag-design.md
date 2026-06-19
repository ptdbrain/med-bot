# Real-Model RAG Design

## Goal

Replace the current mock generation and lexical-only dense branch with an
optional production path using:

- `BAAI/bge-m3` for dense embeddings.
- Qdrant for dense-vector persistence and search.
- `qwen3:4b` served by Ollama for grounded answer generation.
- Existing BM25 plus reciprocal-rank fusion for hybrid retrieval.

The existing mock and lexical path remains the default so tests and offline
development continue to work without downloading models or starting services.

## Configuration

The backend selects providers through environment variables:

```text
LLM_PROVIDER=mock|ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_TIMEOUT_SECONDS=120

EMBEDDING_PROVIDER=lexical|sentence-transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=8

VECTOR_STORE_PROVIDER=memory|qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=medtutor_chunks
```

Valid production combinations require `sentence-transformers` with `qdrant`.
Invalid provider names or a Qdrant configuration without a dense embedding
provider fail at startup with an actionable message.

## Components

### Embedding provider

`app/embeddings/base.py` defines the provider contract. The
sentence-transformers adapter lazily loads BGE-M3 on first embedding request,
normalizes vectors for cosine search, and exposes separate document/query
methods.

### Vector store

The existing in-memory lexical store remains unchanged for offline mode. A new
Qdrant HTTP adapter:

- Creates the configured collection when missing.
- Uses deterministic UUIDs derived from chunk IDs.
- Stores complete chunk text and metadata in point payloads.
- Reconstructs `SearchResult` objects from Qdrant results.
- Uses Qdrant's cosine score threshold directly.

The adapter uses HTTP rather than leaking Qdrant client objects across the
application boundary.

### LLM provider

`OllamaLLM` calls `/api/chat` synchronously with `stream=false`, `think=false`,
and low temperature. The prompt:

- Restricts answers to supplied context.
- Requires numbered citations.
- Requires an explicit insufficient-context answer.
- States the medical-education boundary.

Medical emergency and diagnosis/prescription checks remain ahead of retrieval,
and the service layer still appends the mandatory disclaimer.

### Service assembly

Provider factories construct adapters from `Settings`. `RAGService` receives
optional providers for tests but otherwise uses the factories. BM25 is rebuilt
from SQLite metadata rather than depending on a vector-store-specific `chunks`
property.

## Data Flow

```text
ingest
  -> chunk
  -> SQLite metadata
  -> BGE-M3 document embeddings
  -> Qdrant points

question
  -> safety guardrail
  -> BGE-M3 query embedding -> Qdrant dense candidates
  -> BM25 sparse candidates
  -> reciprocal rank fusion
  -> qwen3:4b with retrieved chunks
  -> cited answer + mandatory disclaimer
```

## Error Handling

- Missing optional embedding dependency: installation command is included in
  the exception.
- Ollama unavailable or model missing: adapter reports URL/model and preserves
  the underlying HTTP status.
- Qdrant unavailable: startup or operation fails with the configured URL.
- No relevant chunks: the current deterministic insufficient-context response
  is returned without calling Ollama.

## Verification

- Unit tests use fake HTTP sessions and fake embedding providers.
- Existing lexical/mock tests must remain green.
- A local smoke test verifies Ollama tags and Qdrant health when those services
  are available.
- Model download and semantic-quality evaluation are separate runtime steps;
  they are not hidden inside unit tests.

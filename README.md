# MedTutor Multimodal RAG Copilot

MedTutor is a portfolio-ready Medical Education RAG assistant. It ingests learning documents, retrieves relevant chunks, answers in Vietnamese with citations, and applies medical-safety guardrails.

This project is not a diagnostic bot, not a prescribing tool, and not a replacement for a clinician. Every medical answer must include:

> Thông tin chỉ phục vụ học tập/tham khảo, không thay thế tư vấn của bác sĩ.

Current quality status and known blockers are recorded in
[`docs/quality/quality-report.md`](docs/quality/quality-report.md). The included
evaluation set is synthetic and unreviewed; it is not a medical accuracy
benchmark.

For a Vietnamese interview-style explanation of the model architecture,
chunking, hybrid retrieval, reranking, evaluation, and troubleshooting, see
[`docs/RAG_INTERVIEW_VI.md`](docs/RAG_INTERVIEW_VI.md).

## Problem

Medical learners often have scattered notes, PDFs, guidelines, slides, images, and lectures. MedTutor provides a clean backend and demo UI for asking questions against those materials while keeping the system inside a safe educational scope.

## Current Scope

Implemented now:

- Phase 1 naive text RAG.
- Phase 2 hybrid retrieval using local vector similarity, Okapi BM25, and Reciprocal Rank Fusion.
- Phase 3 reranking with a deterministic lexical reranker interface.
- Phase 4 parent-child retrieval with persisted parent sections.
- Phase 5 multi-query retrieval and RAG-Fusion.
- Phase 9 rule-based agentic routing.
- Phase 10 corrective retrieval with one query rewrite/retry.
- Phase 11 incremental public-URL updates using persisted content hashes.
- Phase 12 retrieval, answer, and safety evaluation with per-case audit details.
- Optional real-model path using BGE-M3, Qdrant, and Qwen3 through Ollama.
- Ingest `.txt`, `.md`, `.html`, `.pdf`, `.docx`.
- Chunk with document metadata, section title, page marker support, timestamp, and hash.
- Local vector-store fallback that works without Qdrant or embedding API keys.
- SQLite metadata store.
- Mock LLM provider that produces deterministic Vietnamese answers.
- Citations/sources in every grounded answer.
- Shared relevance gate that rejects dense nearest neighbors without a matching topic anchor.
- Guardrails for emergency symptoms and diagnosis/prescription requests.
- Audio metadata validation, configurable STT providers, voice questions, and searchable lecture transcripts.
- Provider-based image OCR/caption boundary with a deterministic mock default.
- Audio questions, searchable audio lectures, and searchable video transcripts.
- Document library API with listing, chunk counts, and index-aware deletion.
- Page-level PDF extraction reports, OCR adapter, stable document/revision IDs, and staged index publication.
- Token-budget chunking option, persistent ingestion jobs, hybrid retrieval diagnostics, and answer/citation validation.
- Per-PDF supervised OCR worker process to isolate Paddle/PyTorch DLL loading; it reuses the model across pages and restarts at a bounded 64-page batch.
- Functional Next.js research workspace plus a compatible Streamlit demo.
- FastAPI backend, sample data, Docker files, and tests.

Supported RAG modes are `naive`, `hybrid`, `rerank`, `parent_child`,
`multi_query`, `corrective`, and `agentic`.

## Safety Scope

MedTutor is for medical learning only:

- No diagnosis.
- No prescriptions.
- No personal treatment decisions.
- Emergency symptoms such as chest pain, stroke signs, severe shortness of breath, loss of consciousness, heavy bleeding, seizures, or suicidal intent trigger an immediate recommendation to contact emergency care.
- The repository uses only fake/sample educational content.

## Architecture

```text
app/main.py
  FastAPI app factory and router registration

app/api/
  HTTP endpoints for health, ingest, chat, image, voice, and evaluation

app/ingestion/
  File loaders, audio/STT processing, mock image processing, chunking, and ingestion pipeline

app/retrieval/
  Local vector fallback, Qdrant dense store, SQLite metadata, BM25 and RRF

app/embeddings/
  Embedding provider interface and lazy sentence-transformers adapter

app/rag/
  Naive, hybrid, rerank, parent-child, multi-query, corrective, and agentic routing

app/llm/
  LLM provider interface, mock provider, and Ollama adapter

app/safety/
  Emergency detection, medical guardrails, policy text, PII redaction

frontend/
  Streamlit demo

web/
  Primary Next.js research workspace for chat, ingest, documents, and evaluation

data/sample/
  Fake medical education sample documents
```

## RAG Methods

Implemented:

- `naive`: retrieve top-k chunks from local vector fallback, generate answer from context, return sources.
- `hybrid`: retrieve candidates from local vector similarity and Okapi BM25, merge them with Reciprocal Rank Fusion, then generate an answer with sources.
- `rerank`: retrieve up to 20 hybrid candidates, rerank them, then generate from the selected top-k.
- `parent_child`: search child chunks and expand the matching persisted section before generation.
- `multi_query`: create deterministic query variants, retrieve each variant, and combine results with RRF.
- `corrective`: validate token relevance, rewrite once when context is weak, and return the insufficient-context response if retry still fails.
- `agentic`: route emergency requests to safety first, comparisons to multi-query, image-related questions to hybrid retrieval, and simple definitions to naive retrieval.

When real-model mode is enabled, the dense branch uses BGE-M3 vectors from
Qdrant while the sparse branch remains BM25.

Before generation, every mode filters candidates against meaningful query
terms. If no topic anchor remains, the service returns the insufficient-context
message instead of sending unrelated nearest neighbors to the LLM.

## Advanced RAG Pipeline

The production-oriented retrieval path is:

```text
structured document
  -> sentence-aware overlapping chunks
  -> contextual indexing (title + section + chunk)
  -> BGE-M3 dense retrieval + BM25 sparse retrieval
  -> weighted reciprocal rank fusion
  -> lexical or cross-encoder reranking
  -> diversity selection
  -> relevance guard
  -> grounded generation with citations
```

Chunking preserves page and Markdown heading boundaries, prefers paragraph and
sentence endings, aligns overlap to text boundaries, and avoids tiny tail
chunks. Each chunk records `chunk_index`, `char_start`, and `char_end` in
metadata. Defaults are configurable:

```env
CHUNK_SIZE=900
CHUNK_OVERLAP=120
CHUNK_MIN_SIZE=180
```

Dense and sparse candidate lists are fused with weighted RRF. Increase the
dense weight for multilingual or paraphrased questions; increase the sparse
weight for exact terminology, drug names, and laboratory markers:

```env
HYBRID_CANDIDATE_MULTIPLIER=4
HYBRID_RRF_K=60
HYBRID_DENSE_WEIGHT=1.0
HYBRID_SPARSE_WEIGHT=1.0
```

The default reranker is deterministic and combines initial retrieval score,
query-term coverage, title/section coverage, exact phrase evidence, and
MMR-style diversity. For multilingual semantic reranking, enable the
cross-encoder adapter:

```env
RERANKER_PROVIDER=cross-encoder
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_CANDIDATE_K=20
RERANKER_BATCH_SIZE=8
RERANKER_DIVERSITY_LAMBDA=0.75
```

`GET /retrieve` includes a `signals` object for diagnostics. Depending on the
mode and providers it contains dense/sparse ranks and scores, lexical coverage,
title coverage, phrase match, reranker logit, and normalized semantic score.

Changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `CHUNK_MIN_SIZE` requires ingesting
the source document again because chunk boundaries are persisted in SQLite.
Re-ingest atomically replaces the previous chunks for the same document and
removes stale vector entries.

For local semantic retrieval without Docker, use the SQLite dense adapter:

```env
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
VECTOR_STORE_PROVIDER=sqlite-dense
DENSE_INDEX_PATH=data/indexes/dense_vectors.db
RERANKER_PROVIDER=lexical
```

`sqlite-dense` stores float32 vectors and chunk payloads in a local SQLite
database. It is intended for development and small/medium corpora. Qdrant
remains the recommended provider for concurrent or production deployments.

## Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The default `STT_PROVIDER=mock` needs no model download. For real local transcription:

```bash
pip install -r requirements-audio.txt
```

Then set:

```env
STT_PROVIDER=faster-whisper
WHISPER_MODEL_SIZE=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
STT_LANGUAGE=vi
```

Run the backend:

```bash
uvicorn app.main:app --reload
```

Run the Streamlit demo:

```bash
streamlit run frontend/streamlit_app.py
```

Backend default: `http://localhost:8000`  
Next.js frontend: `http://localhost:3000`
Legacy Streamlit frontend: `http://localhost:8501`

For profile-specific setup, readiness checks, queued ingestion, recovery, and
evaluation caveats, use [`docs/runbooks/rag-local.md`](docs/runbooks/rag-local.md).
The default `demo` profile is deterministic and does not prove semantic model
quality. `local` opts into Ollama, BGE-M3, Qdrant, PaddleOCR, and the
cross-encoder; these dependencies and model weights must be installed and
available. `GET /ready` checks configured providers, while `GET /health` only
reports process liveness and index counts.

## Real-Model RAG

Recommended local configuration:

```text
Embedding: BAAI/bge-m3
Dense store: Qdrant
Sparse retrieval: BM25
Fusion: Reciprocal Rank Fusion
Generator: qwen3:4b-instruct-2507-q4_K_M through Ollama
```

Qdrant is optional for local development. Replace it with
`VECTOR_STORE_PROVIDER=sqlite-dense` when Docker is unavailable.

### 1. Start Ollama on Windows

Ollama is installed outside WSL. Open PowerShell:

```powershell
ollama serve
```

Open another PowerShell window:

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama list
```

Ollama must respond at `http://localhost:11434`.

### 2. Start Qdrant

From the project root:

```bash
docker compose up -d qdrant
curl http://localhost:6333/healthz
```

### 3. Install embedding dependencies

```bash
source .venv/bin/activate
pip install -r requirements-rag.txt
```

This installs the CPU-only PyTorch wheel. The first ingest or search downloads
`BAAI/bge-m3` from Hugging Face.

On Windows without Developer Mode, Hugging Face may fail to create cache
symlinks with `WinError 1314`. Download directly into the project data
directory instead:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='data/models/bge-m3')"
```

Then configure:

```env
EMBEDDING_MODEL=data/models/bge-m3
```

`data/models` is ignored by Git and mounted through the existing Docker
`data:/app/data` volume.

### 4. Enable real-model mode

Copy `.env.example` to `.env`, then set:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M
OLLAMA_TIMEOUT_SECONDS=180
OLLAMA_THINK=false
OLLAMA_NUM_PREDICT=512

EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=8

VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=medtutor_chunks

RERANKER_PROVIDER=cross-encoder
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_CANDIDATE_K=20
```

Use `rag_mode=rerank` to apply the configured reranker. `rag_mode=hybrid`
stops after weighted dense/BM25 fusion.

For a lower-memory first run, keep `RERANKER_PROVIDER=lexical`. This loads only
BGE-M3 for dense retrieval. Enable the cross-encoder after validating dense
retrieval quality and available memory.

Keep `OLLAMA_THINK=false` for normal grounded RAG on CPU. Set it to `true` only
when the extra reasoning latency is acceptable; long contexts can otherwise
exceed `OLLAMA_TIMEOUT_SECONDS`. `OLLAMA_NUM_PREDICT` bounds answer length and
keeps local CPU generation latency predictable.

Start FastAPI with the environment file:

```bash
uvicorn app.main:app --reload --env-file .env --port 8000
```

Start the Next.js frontend in another terminal:

```bash
cd web
npm run dev
```

Open `http://localhost:3000/chat`.

### 5. Ingest before asking questions

```bash
set -a
source .env
set +a
python -m scripts.ingest_sample_data
```

Use `hybrid` mode in the chat interface. The request path is:

```text
query -> BGE-M3/Qdrant + BM25 -> RRF
      -> qwen3:4b-instruct-2507-q4_K_M -> cited answer
```

If chunks already exist in SQLite but Qdrant is new or was reset, rebuild the
dense index without re-ingesting source files:

```bash
python -m scripts.reindex_vectors --clear --batch-size 32
```

The command commits every batch. If it is interrupted, continue without
re-encoding completed chunks:

```bash
python -m scripts.reindex_vectors --resume --batch-size 32
```

## Docker

```bash
docker compose up --build
```

To include `faster-whisper` and BGE-M3 dependencies in the Docker image:

```bash
INSTALL_AUDIO=true INSTALL_RAG=true docker compose up --build
```

Services:

- Backend: `http://localhost:8000`
- Legacy Streamlit frontend: `http://localhost:8501`
- Qdrant: `http://localhost:6333`

The primary Next.js frontend is run separately from `web/` and is available at
`http://localhost:3000`.

Mock/lexical mode remains the default. Set `INSTALL_RAG=true`,
`EMBEDDING_PROVIDER=sentence-transformers`, and either
`VECTOR_STORE_PROVIDER=sqlite-dense` or `VECTOR_STORE_PROVIDER=qdrant` to
enable the dense path in Docker.

## Ingest Sample Data

```bash
python -m scripts.ingest_sample_data
```

Sample files cover:

- Hypertension learning notes.
- Stroke/FAST warning signs.
- Blood-test indicators.
- Emergency warning examples.
- Medicine safety and contraindications at learning level.

## Chat Usage

API example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"FAST trong dot quy gom nhung dau hieu nao?","rag_mode":"naive","top_k":5}'
```

Use hybrid retrieval:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"FAST trong dot quy gom nhung dau hieu nao?","rag_mode":"hybrid","top_k":5}'
```

Expected response fields:

- `answer`
- `sources`
- `retrieved_chunks`
- `mode_used`
- `safety_warning`

If context is missing, the answer says:

```text
Tôi chưa tìm thấy thông tin này trong tài liệu đã được cung cấp.
```

## Audio Processing

Supported extensions:

- `.wav`
- `.mp3`
- `.m4a`
- `.ogg`

Transcribe an audio file:

```bash
curl -X POST http://localhost:8000/voice/transcribe \
  -F "file=@question.wav"
```

Ask RAG using an audio question:

```bash
curl -X POST "http://localhost:8000/voice/chat?rag_mode=hybrid&top_k=5" \
  -F "file=@question.wav"
```

Ingest an audio lecture so its transcript becomes searchable:

```bash
curl -X POST http://localhost:8000/voice/ingest \
  -F "file=@lecture.wav"
```

The audio pipeline is:

```text
upload validation
  -> audio metadata inspection
  -> configured STT provider
  -> transcript and timestamped segments
  -> question RAG or lecture chunking/indexing
```

WAV files are validated and include duration, sample rate, channel count, and sample width. Other formats rely on the configured STT provider for decoding.

## Image And Video Processing

Ingest an educational image:

```bash
curl -X POST http://localhost:8000/image/ingest \
  -F "file=@infographic.png"
```

The image processor uses separate OCR and caption provider interfaces. The
default implementation is deterministic and intended for local demos; a real
OCR or vision provider can replace it without changing API routes.

Ingest a lecture video:

```bash
curl -X POST http://localhost:8000/video/ingest \
  -F "file=@lecture.mp4"
```

Supported video extensions are `.mp4`, `.mov`, `.mkv`, and `.webm`. The current
MVP sends the uploaded media to the configured STT provider and indexes the
resulting transcript. Keyframe extraction is intentionally not part of the
basic path.

## Document Library And URL Updates

List indexed documents:

```bash
curl http://localhost:8000/documents
```

Delete one document, its raw upload, and its chunks from SQLite, BM25, and the
active vector store:

```bash
curl -X DELETE http://localhost:8000/documents/<doc_id>
```

`POST /ingest/url` only accepts public HTTP/HTTPS sources. The backend stores a
content hash per URL, skips unchanged pages, and replaces the previous indexed
document when content changes.

## Evaluation

The `evaluation/` corpus and cases are assistant-authored synthetic drafts.
Use `--allow-draft` only for pipeline development; reports must remain marked
`not_evaluated_unreviewed` until qualified reviewers approve the evidence and
answers. See the protocol and current blockers in `docs/quality/`.

Run:

```bash
python -m scripts.run_eval
```

The evaluation dataset is at `app/evaluation/eval_dataset.json`.

Metrics included:

- `hit_at_k`
- `mrr`
- `citation_rate`
- `format_compliance_rate`
- `disclaimer_rate`
- `insufficient_context_accuracy`
- `safety_accuracy`

`POST /eval/run` also returns per-question details for retrieval rank, citation,
insufficient-context behavior, disclaimer presence, and the actual safety
warning. The Next.js Evaluation page displays these checks as an audit table.
`citation_rate` requires both a retrieved source and an actual `[n]` citation
in the answer. `format_compliance_rate` requires either all four configured
headings or four numbered answer blocks. Both are measured only on answerable
cases; safety-blocked and insufficient-context cases are expected to have no
citations.

Older smoke snapshots below this point were produced before the current
quality pipeline and are not release evidence. Use the current report instead.

## Tests

```bash
python -m pytest tests -q
```

Minimum tests cover:

- Sentence-aware chunking, overlap, tail handling, and required metadata.
- Emergency detection.
- Diagnosis/prescription guardrails.
- Local vector retrieval, Okapi BM25, and weighted Reciprocal Rank Fusion.
- Multi-signal lexical reranking, diversity selection, and cross-encoder adapter.
- Cross-language semantic relevance signals.
- WAV validation, STT provider selection, voice chat, and searchable lecture ingestion.
- Chat API answer and sources.
- Empty-context answer.

## API Endpoints

- `GET /health`
- `GET /ready`
- `POST /ingest/file`
- `POST /ingest/jobs` (queued upload, returns HTTP 202)
- `GET /ingest/jobs/{job_id}`
- `POST /ingest/jobs/{job_id}/cancel`
- `POST /ingest/jobs/{job_id}/retry`
- `POST /ingest/url`
- `GET /documents`
- `DELETE /documents/{doc_id}`
- `POST /chat`
- `GET /retrieve`
- `POST /voice/transcribe`
- `POST /voice/chat`
- `POST /voice/ingest`
- `POST /image/ingest`
- `POST /video/ingest`
- `POST /eval/run`

## Demo Script

1. Start backend and frontend.
2. Run `python -m scripts.ingest_sample_data`.
3. Open the Next.js workspace at `http://localhost:3000/chat`.
4. Ask: `FAST trong đột quỵ gồm những dấu hiệu nào?`
5. Confirm the answer has citations and the medical disclaimer.
6. Ask: `Hãy chẩn đoán bệnh và kê đơn thuốc cho tôi`.
7. Confirm the guardrail blocks diagnosis/prescription.
8. Ask: `Tôi bị đau ngực và khó thở, cần làm gì?`
9. Confirm the emergency response appears before retrieval.

## Portfolio Highlights

- Modular RAG architecture with provider interfaces.
- Offline demo mode with no API keys.
- Medical safety guardrails are first-class, not an afterthought.
- Citation-aware answer format.
- Local persistence with a clear path to Qdrant/PostgreSQL.
- Testable FastAPI service and Streamlit demo.

## Roadmap

1. Build a labeled retrieval evaluation set for the collected corpus.
2. Measure BGE-M3 + BM25 hybrid retrieval against the lexical baseline.
3. Tune dense/sparse weights and reranker thresholds on the labeled corpus.
4. Replace mock OCR/caption defaults with tested local or hosted providers.
5. Add keyframe extraction for video when transcript-only evaluation shows a gap.
7. Add APScheduler or cron deployment around the existing manual URL updater.
8. Add answer-faithfulness and citation-entailment evaluation.
9. Add authentication before exposing uploads outside a local/private environment.

## Current Limitations

- Real embedding and Qdrant are optional; the default remains deterministic lexical matching.
- BGE-M3 model download and CPU inference require additional disk, RAM, and startup time.
- PDF/DOCX ingestion requires optional packages from `requirements.txt`.
- The current PaddleOCR Vietnamese live fixture lost diacritics despite high confidence; scan-PDF quality is not accepted yet.
- The conservative OCR render default is 160 DPI with a 2-million-pixel inference cap because the 250-DPI Windows worker exited natively in live checks. The OCR worker now persists for up to 64 pages per PDF; live Vietnamese fidelity remains unverified and the OCR quality gate is still FAIL.
- Human-reviewed support labels and an audited held-out medical-education test set do not exist yet.
- Live Ollama, Qdrant, complete UI E2E, cloud gateway, and 10k/50k scale tests were not run for this implementation snapshot.
- Image captioning remains outside the OCR provider contract; demo image processing is not evidence of multimodal semantic understanding.
- Audio defaults to mock STT unless the optional `faster-whisper` dependency and provider setting are enabled.
- Video is transcript-only and depends on the configured STT provider.
- The rule-based query transformer and lexical reranker are deterministic baselines, not learned models; the optional cross-encoder path requires its model and live evaluation.
- The mock LLM summarizes retrieved chunks; it is not a clinical reasoning model.

## Medical Use Warning

Do not use this system for diagnosis, prescriptions, triage decisions, or treatment planning. It is a learning assistant for supplied educational documents only.

# MedTutor Multimodal RAG Copilot

MedTutor is a portfolio-ready Medical Education RAG assistant. It ingests learning documents, retrieves relevant chunks, answers in Vietnamese with citations, and applies medical-safety guardrails.

This project is not a diagnostic bot, not a prescribing tool, and not a replacement for a clinician. Every medical answer must include:

> Thông tin chỉ phục vụ học tập/tham khảo, không thay thế tư vấn của bác sĩ.

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

## Real-Model RAG

Recommended local configuration:

```text
Embedding: BAAI/bge-m3
Dense store: Qdrant
Sparse retrieval: BM25
Fusion: Reciprocal Rank Fusion
Generator: qwen3:4b through Ollama
```

### 1. Start Ollama on Windows

Ollama is installed outside WSL. Open PowerShell:

```powershell
ollama serve
```

Open another PowerShell window:

```powershell
ollama pull qwen3:4b
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

### 4. Enable real-model mode

Copy `.env.example` to `.env`, then set:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_TIMEOUT_SECONDS=180

EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=8

VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=medtutor_chunks
```

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
query -> BGE-M3/Qdrant + BM25 -> RRF -> qwen3:4b -> cited answer
```

If chunks already exist in SQLite but Qdrant is new or was reset, rebuild the
dense index without re-ingesting source files:

```bash
python -m scripts.reindex_vectors --clear
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
`EMBEDDING_PROVIDER=sentence-transformers`, and
`VECTOR_STORE_PROVIDER=qdrant` to enable the dense path in Docker.

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

Run:

```bash
python -m scripts.run_eval
```

The evaluation dataset is at `app/evaluation/eval_dataset.json`.

Metrics included:

- `hit_at_k`
- `mrr`
- `citation_rate`
- `disclaimer_rate`
- `insufficient_context_accuracy`
- `safety_accuracy`

`POST /eval/run` also returns per-question details for retrieval rank, citation,
insufficient-context behavior, disclaimer presence, and the actual safety
warning. The Next.js Evaluation page displays these checks as an audit table.
`citation_rate` is measured only on answerable cases; safety-blocked and
insufficient-context cases are expected to have no citations.

## Tests

```bash
python -m pytest tests -q
```

Minimum tests cover:

- Chunking and required metadata.
- Emergency detection.
- Diagnosis/prescription guardrails.
- Local vector retrieval, Okapi BM25, and Reciprocal Rank Fusion.
- WAV validation, STT provider selection, voice chat, and searchable lecture ingestion.
- Chat API answer and sources.
- Empty-context answer.

## API Endpoints

- `GET /health`
- `POST /ingest/file`
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
3. Add reranking through a cross-encoder provider only if evaluation justifies it.
4. Replace mock OCR/caption defaults with tested local or hosted providers.
5. Add keyframe extraction for video when transcript-only evaluation shows a gap.
6. Add a cross-encoder reranker adapter and compare it with the lexical baseline.
7. Add APScheduler or cron deployment around the existing manual URL updater.
8. Add answer-faithfulness and citation-entailment evaluation.
9. Add authentication before exposing uploads outside a local/private environment.

## Current Limitations

- Real embedding and Qdrant are optional; the default remains deterministic lexical matching.
- BGE-M3 model download and CPU inference require additional disk, RAM, and startup time.
- PDF/DOCX ingestion requires optional packages from `requirements.txt`.
- Image OCR/caption defaults are still mock providers.
- Audio defaults to mock STT unless the optional `faster-whisper` dependency and provider setting are enabled.
- Video is transcript-only and depends on the configured STT provider.
- The rule-based query transformer and reranker are deterministic baselines, not learned models.
- The mock LLM summarizes retrieved chunks; it is not a clinical reasoning model.

## Medical Use Warning

Do not use this system for diagnosis, prescriptions, triage decisions, or treatment planning. It is a learning assistant for supplied educational documents only.

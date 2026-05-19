# MedTutor Multimodal RAG Copilot

MedTutor is a portfolio-ready Medical Education RAG assistant. It ingests learning documents, retrieves relevant chunks, answers in Vietnamese with citations, and applies medical-safety guardrails.

This project is not a diagnostic bot, not a prescribing tool, and not a replacement for a clinician. Every medical answer must include:

> Thông tin chỉ phục vụ học tập/tham khảo, không thay thế tư vấn của bác sĩ.

## Problem

Medical learners often have scattered notes, PDFs, guidelines, slides, images, and lectures. MedTutor provides a clean backend and demo UI for asking questions against those materials while keeping the system inside a safe educational scope.

## Current Scope

Implemented now:

- Phase 1 naive text RAG.
- Ingest `.txt`, `.md`, `.html`, `.pdf`, `.docx`.
- Chunk with document metadata, section title, page marker support, timestamp, and hash.
- Local vector-store fallback that works without Qdrant or embedding API keys.
- SQLite metadata store.
- Mock LLM provider that produces deterministic Vietnamese answers.
- Citations/sources in every grounded answer.
- Guardrails for emergency symptoms and diagnosis/prescription requests.
- Mock image OCR/caption and mock audio STT boundaries.
- FastAPI backend, Streamlit demo, sample data, Docker files, and tests.

Advanced RAG modes are scaffolded but intentionally not presented as complete yet. The MVP returns an explicit message if a non-`naive` mode is requested.

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
  File loaders, mock media processors, chunking, and ingestion pipeline

app/retrieval/
  Local vector fallback, SQLite metadata, BM25/reranker/query-transform boundaries

app/rag/
  Naive RAG service plus placeholders for later RAG modes

app/llm/
  LLM provider interface, mock provider, future OpenAI/local providers

app/safety/
  Emergency detection, medical guardrails, policy text, PII redaction

frontend/
  Streamlit demo

data/sample/
  Fake medical education sample documents
```

## RAG Methods

Implemented:

- `naive`: retrieve top-k chunks from local vector fallback, generate answer from context, return sources.

Scaffolded for later phases:

- `hybrid`
- `rerank`
- `parent_child`
- `multi_query`
- `corrective`
- `agentic`

## Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
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
Frontend default: `http://localhost:8501`

## Docker

```bash
docker compose up --build
```

Services:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`
- Qdrant: `http://localhost:6333`

Phase 1 uses local vector fallback by default, so Qdrant is available for later integration but not required for the MVP path.

## Ingest Sample Data

```bash
python scripts/ingest_sample_data.py
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

## Evaluation

Run:

```bash
python scripts/run_eval.py
```

The evaluation dataset is at `app/evaluation/eval_dataset.json`.

Metrics included:

- `hit_at_k`
- `mrr`
- `citation_rate`
- `disclaimer_rate`
- `insufficient_context_accuracy`
- `safety_accuracy`

## Tests

```bash
python -m pytest tests -q
```

Minimum tests cover:

- Chunking and required metadata.
- Emergency detection.
- Diagnosis/prescription guardrails.
- Local retrieval.
- Chat API answer and sources.
- Empty-context answer.

## API Endpoints

- `GET /health`
- `POST /ingest/file`
- `POST /ingest/url`
- `POST /chat`
- `GET /retrieve`
- `POST /voice/transcribe`
- `POST /voice/chat`
- `POST /voice/ingest`
- `POST /image/ingest`
- `POST /eval/run`

## Demo Script

1. Start backend and frontend.
2. Run `python scripts/ingest_sample_data.py`.
3. Open Streamlit.
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

1. Replace lexical vector fallback with sentence-transformers embeddings and Qdrant.
2. Add true hybrid dense + BM25 search with Reciprocal Rank Fusion.
3. Add reranking through a cross-encoder provider.
4. Implement parent-child RAG for long PDFs/guidelines.
5. Add multi-query and RAG-Fusion.
6. Replace mock OCR/STT/captioner with provider adapters.
7. Add video transcript ingestion with ffmpeg/moviepy.
8. Add rule-based then LLM-assisted agentic routing.
9. Add corrective retrieval and stronger answer faithfulness evaluation.
10. Add scheduled public-URL update jobs.

## Current Limitations

- No real embedding model is used in Phase 1; retrieval is deterministic lexical matching.
- Qdrant is included in Docker but not wired into the default retrieval path yet.
- PDF/DOCX ingestion requires optional packages from `requirements.txt`.
- Image and audio support are provider-boundary mocks in this phase.
- The mock LLM summarizes retrieved chunks; it is not a clinical reasoning model.

## Medical Use Warning

Do not use this system for diagnosis, prescriptions, triage decisions, or treatment planning. It is a learning assistant for supplied educational documents only.

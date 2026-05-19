# AGENTS.md

Guidance for coding agents working on MedTutor Multimodal RAG Copilot.

- Keep the modular architecture. Do not collapse ingestion, retrieval, safety, LLM, API, and UI logic into one file.
- Run relevant tests for every change. When adding a RAG mode, add or update tests and README sections.
- Prefer clear deterministic code over complex framework magic.
- Provider integrations must go through an interface or adapter. Do not hard-code OpenAI, Qdrant, OCR, STT, or vision-provider calls directly inside API routes.
- Do not commit `.env`, API keys, real patient records, or sensitive medical data.
- The product is a medical education assistant, not a diagnostic or prescribing system.
- Every medical answer must preserve the disclaimer: `Thông tin chỉ phục vụ học tập/tham khảo, không thay thế tư vấn của bác sĩ.`
- Emergency symptoms must trigger an emergency/care-seeking response before retrieval.
- If retrieval has no relevant context, answer that the provided documents do not contain enough information. Do not invent citations.

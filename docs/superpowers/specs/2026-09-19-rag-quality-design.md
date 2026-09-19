# RAG Quality Completion Design

## 1. Objective

Complete MedTutor's core RAG path so that text PDFs, scanned PDFs, Vietnamese
and English documents can be ingested, chunked, retrieved, and answered with
measurable quality. The first deployment remains local-first, while OCR,
embedding, reranking, vector storage, and generation stay replaceable through
provider interfaces for later cloud or hybrid operation.

This work prioritizes ingestion quality, chunk quality, retrieval relevance,
grounded answers, and evaluation. Public deployment, authentication, billing,
and unrelated UI redesign are outside this scope.

## 2. Current Baseline

The repository already provides:

- FastAPI, Next.js, SQLite metadata, SQLite dense vectors, and optional Qdrant.
- BGE-M3 dense embeddings, BM25, weighted RRF, lexical/cross-encoder reranking,
  parent-child, multi-query, corrective, and agentic modes.
- Medical emergency and prohibited-advice guardrails.
- Re-ingestion, vector cleanup, citation-aware generation, and a small eval set.
- A local corpus with 4,137 metadata chunks and matching dense vectors.

The main gaps are:

- PDF extraction has no page-level scan detection or OCR fallback.
- Extraction quality and OCR confidence are not measured or surfaced.
- Chunk sizes use characters rather than model tokens and do not explicitly
  preserve tables or lists.
- Chunk/version metadata is incomplete for safe future re-indexing.
- Relevance thresholds are hard-coded and not tuned on a labeled corpus.
- The evaluation set is too small to support chunk/retrieval decisions.
- Citation presence is measured, but citation entailment and answer
  faithfulness are not yet measured.

## 3. Architecture

Use one provider-driven pipeline with three runtime profiles:

- `local`: local OCR, BGE-M3, local cross-encoder, Qdrant or SQLite dense,
  and Ollama.
- `cloud`: hosted OCR, embedding, reranking, and LLM adapters.
- `hybrid`: any supported combination of local and cloud providers.

All profiles share the same `Document`, `Chunk`, `SearchResult`, ingestion
report, and evaluation contracts. API routes call services only; they do not
call OCR, embedding, vector database, reranker, or LLM providers directly.

```text
file
  -> file/page classification
  -> text extraction or OCR
  -> normalization
  -> structure-aware token chunking
  -> chunk quality gate
  -> metadata + dense index + BM25

question
  -> Unicode/language normalization
  -> dense + BM25 candidates
  -> weighted RRF
  -> multilingual cross-encoder reranking
  -> deduplication/diversity
  -> calibrated relevance gate
  -> parent expansion when needed
  -> grounded generation
  -> citation/faithfulness/safety validation
```

## 4. Ingestion Design

### 4.1 Page-level PDF classification

Each PDF page is classified independently:

- `text`: extracted text passes quality thresholds.
- `scan`: extracted text is empty or below quality thresholds and the page
  contains renderable visual content.
- `mixed`: extracted text exists but is incomplete or low quality.

Text pages use `pypdf`. Scan pages are rendered to images and sent through an
`OCRProvider`. Mixed pages use extracted text when it passes quality checks and
OCR otherwise. The system must not OCR every page when the text layer is good.

The default local OCR provider is PaddleOCR because Vietnamese and English are
both required. A deterministic fake provider remains available for tests. A
cloud OCR adapter can be added without changing `PdfLoader` or API routes.

### 4.2 Extraction quality

For every page, record:

- extracted character and word counts;
- printable-character ratio;
- replacement/control-character ratio;
- repeated-line ratio;
- selected extraction method;
- OCR confidence when OCR is used;
- warnings such as `empty_page`, `low_text_quality`, or
  `low_ocr_confidence`.

The ingest response includes totals for processed, text, OCR, warning, and
skipped pages. A document with no usable pages fails ingestion and must not
replace its previously valid index.

### 4.3 Normalization

Normalization removes repeated headers and footers, repairs line-wrapped
sentences and conservative hyphenation, normalizes Unicode and whitespace,
and preserves headings, lists, tables, captions, and page boundaries. It must
not ask an LLM to rewrite medical terminology.

## 5. Chunking Design

Chunk in this order:

1. document;
2. page;
3. heading or section;
4. paragraph, table, or list block;
5. sentence-aware token windows.

Default targets are 500 tokens, 80 tokens overlap, and 100 minimum tokens.
These are configuration defaults, not universal truth; evaluation determines
whether they change. Tables and lists remain atomic unless they exceed the
hard maximum. Oversized blocks split by row/item before sentence fallback.

Each child chunk has a larger parent section for parent-child retrieval. Chunk
metadata includes:

- `doc_id`, `chunk_id`, `parent_id`, source path/type, title, language;
- page number and section title;
- character and token positions and token count;
- extraction method and OCR confidence;
- document content hash;
- chunker version and embedding version.

The quality gate rejects empty and tiny chunks, identifies exact and
near-duplicate chunks, and reports length distribution and warnings. Re-ingest
replaces metadata and vectors only after all new chunks pass validation.

## 6. Retrieval Design

### 6.1 Query processing

Normalize Unicode and support Vietnamese with or without diacritics while
preserving medical entities, abbreviations, drug names, genes, enzymes, and
laboratory markers. Multi-query is reserved for comparisons or clearly
multi-part questions; simple questions use one query.

### 6.2 Candidate generation and fusion

- Dense retrieval with BGE-M3: default top 40.
- BM25 retrieval: default top 40.
- Weighted RRF combines ranks without assuming score-scale compatibility.
- Diagnostic signals retain dense/sparse rank and score.

### 6.3 Reranking and selection

The production local profile uses a multilingual cross-encoder on the top
20-40 fused candidates. The deterministic lexical-semantic reranker remains a
test and low-resource fallback. MMR-style selection and duplicate detection
prevent overlapping chunks from filling the final context. The final context
contains 5-8 chunks within an explicit token budget.

### 6.4 Relevance gate

Acceptance uses calibrated dense, reranker, lexical-anchor, and rank signals.
Thresholds are configuration values selected on the labeled development set,
not hard-coded constants. If no candidate passes, the system returns the
insufficient-context response and does not call the generator.

Each result records why it was accepted or rejected so poor retrieval can be
diagnosed from `/retrieve` and evaluation details.

## 7. Answer Design

Grounded generation receives immutable source IDs `[1]`, `[2]`, and so on.
The answer contains four sections: concise answer, explanation, sources, and
medical warning. Important claims carry citations. Emergency handling occurs
before retrieval, and the exact medical disclaimer is applied by application
code rather than entrusted only to the model.

Post-generation validation checks:

- every cited ID exists in the supplied context;
- an answerable response has at least one citation;
- the required four-part structure is present;
- claims are supported by cited chunks;
- prohibited diagnosis or prescription language is absent.

One repair attempt is allowed for citation or formatting failures. If
faithfulness still fails, return a controlled insufficient-context response
rather than an unsupported answer.

## 8. Evaluation Design

Build a versioned, labeled evaluation corpus before tuning. The first useful
set contains at least 120 cases:

- 30 exact-keyword questions;
- 30 paraphrase/semantic questions;
- 20 Vietnamese-English cross-language questions;
- 15 comparison or multi-hop questions;
- 15 unanswerable questions;
- 10 safety/emergency/prohibited-advice questions.

Each answerable case records relevant document IDs and, where practical,
relevant chunk or page IDs. Keep a development split for tuning and a held-out
test split for reporting.

Measure:

- ingestion success and extraction warning rates;
- chunk count/length/duplicate distributions;
- Recall@5/10, Hit@5, MRR, and nDCG@10;
- unanswerable precision/recall;
- citation presence and citation correctness;
- answer faithfulness and format compliance;
- disclaimer and safety accuracy;
- p50/p95 latency for ingest, retrieval, rerank, and generation.

Compare at minimum BM25, dense, hybrid, and hybrid plus reranker. Chunk size,
overlap, fusion weights, candidate counts, and relevance thresholds may change
only when the development metrics improve without regressing the held-out set
or safety checks.

## 9. Failure Handling and Observability

- A page extraction failure does not fail the whole document unless no usable
  page remains; warnings are returned with page numbers.
- A failed re-ingest leaves the previously valid document and vectors intact.
- Provider errors include provider name, stage, retryability, and a safe
  message without document content or credentials.
- Structured logs record document ID, stage, duration, counts, model/version,
  and warnings, but never full medical document contents.
- Health output reports active profile/provider names, index counts, model
  versions, and index/config version compatibility.

## 10. Scalability Boundaries

SQLite dense remains the small local-development option. Qdrant is the default
for the scalable profile. Metadata storage stays behind its repository/service
boundary so PostgreSQL can replace SQLite later. Batch embedding, resumable
re-indexing, content hashes, and versioned index metadata prevent full rebuilds
unless chunking or embedding compatibility changes.

## 11. Acceptance Criteria

The work is complete when:

1. Text, scanned, and mixed PDFs produce page-level extraction reports and
   searchable chunks in Vietnamese and English.
2. An all-image PDF is ingested through OCR; a text PDF does not invoke OCR on
   good pages.
3. Chunk metadata contains token positions/counts, extraction provenance, and
   version fields; duplicate and tiny chunks are rejected or reported.
4. Failed ingestion cannot destroy a previously valid index.
5. Hybrid plus reranker beats or matches both BM25 and dense baselines on
   Recall@10 and MRR on the held-out evaluation set.
6. Unanswerable precision and safety accuracy are each at least 0.95 on the
   held-out set.
7. Citation correctness, disclaimer rate, and answer-format compliance are
   each at least 0.95.
8. Every metric is reproducible through one documented command and produces a
   machine-readable report.
9. Local mode works without paid APIs; provider contract tests demonstrate
   that cloud adapters can be selected without changing API routes.
10. Backend tests, frontend tests, Ruff, Black, frontend lint/build, Docker
    configuration, and end-to-end smoke tests all pass.

## 12. Non-Goals

- Clinical diagnosis, treatment recommendation, or prescribing.
- Public multi-tenant deployment and authentication in this workstream.
- Handwritten-document recognition.
- Training a new embedding model or reranker from scratch.
- Video keyframe understanding unless transcript evaluation demonstrates a
  concrete retrieval gap.

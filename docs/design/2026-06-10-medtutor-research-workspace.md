# MedTutor Research Workspace Design Specification

## Product

MedTutor is a research-oriented medical learning assistant for students. The interface must help users ask questions, inspect evidence, manage trusted source documents, ingest new material, and measure RAG quality without presenting the product as a diagnostic system.

The primary success criterion is not message volume. It is whether a student can understand an answer and verify exactly which document, section, page, and chunk supported it.

## Design Direction

Selected direction: **Research Workspace**.

Physical scene: a medical student studies for several hours on a laptop in a quiet library, moving repeatedly between a question, an explanation, and the original evidence. The interface therefore uses a dark theme to reduce glare, but maintains high text contrast and restrained density.

Anchor references:

- Perplexity for answer-to-source interaction.
- Linear for navigation discipline and compact controls.
- Scientific paper readers for evidence inspection.

Avoid:

- Marketing composition or oversized hero headings.
- Decorative medical imagery.
- Bright hospital-white and generic teal branding.
- JSON shown directly to end users.
- Nested cards, gradients, glass effects, or excessive rounded containers.

## Information Architecture

Routes:

```text
/chat
/chat/[conversationId]
/documents
/documents/[documentId]
/ingest
/evaluation
/settings
```

Persistent left navigation:

1. Chat
2. Kho tài liệu
3. Ingest dữ liệu
4. Đánh giá RAG
5. Cài đặt

The footer of the navigation shows backend connectivity, indexed chunk count, and current corpus.

## Application Shell

Desktop:

```text
220px navigation | flexible workspace | optional 360px evidence inspector
```

Tablet:

- Navigation collapses to an icon rail.
- Evidence inspector becomes a right drawer.

Mobile:

- Navigation becomes a sheet.
- Evidence inspector becomes a full-height sheet.
- Chat composer remains visible above the safe area.

The shell must preserve route state and panel widths. Main content must not horizontally overflow at 320px viewport width.

## Chat Workspace

### Header

- Conversation title.
- Active corpus selector.
- RAG mode segmented control: `Naive`, `Hybrid`.
- Compact settings button for `top_k`.
- Evidence-panel toggle.

Unsupported RAG modes must not appear.

### Conversation

User messages are compact and right-aligned. Assistant answers occupy the readable main column and support:

- Markdown paragraphs and lists.
- Numbered inline citation buttons such as `[1]`.
- Safety warning blocks.
- Insufficient-context state.
- Copy answer action.
- Retry action.
- Display of mode used and elapsed time when available.

Assistant answers must not be placed inside decorative cards. A subtle separator and author row are sufficient.

### Composer

- Multiline input.
- Send icon button.
- Attach document button.
- Microphone button.
- Disabled, recording, uploading, submitting, and error states.
- Short scope reminder below the field.

Enter sends; Shift+Enter adds a line.

### Evidence Inspector

The inspector is synchronized with citations:

- Source number and title.
- Organization and document type when available.
- Page and section.
- Retrieval score.
- Exact retrieved excerpt.
- Open-document action.
- Previous/next source controls.

Selecting `[2]` scrolls or switches the inspector to source 2. Selecting an excerpt in the inspector highlights its citation in the answer.

### Chat States

- First use with no indexed documents.
- Documents exist, but no conversation yet.
- Loading with skeleton answer lines.
- Streaming-ready layout even if the first backend version is non-streaming.
- Backend offline.
- No relevant context.
- Emergency safety response.
- Diagnosis/prescription refusal.
- Citation source missing.

## Documents Workspace

### List

Use a data table on desktop and a structured list on mobile.

Columns:

- Title
- Organization
- Type
- Topic
- Language
- Chunks
- Index status
- Updated

Controls:

- Search.
- Topic, type, language, and status filters.
- Sort.
- Open details.
- Re-index action.

### Document Detail

- Source metadata and original URL.
- File information and content hash.
- Extracted sections/pages.
- Chunk list with search.
- Index state and last ingest run.
- Open original source.

Deletion must require an explicit confirmation and state whether vectors/chunks are also removed.

## Ingest Workspace

Use one continuous workflow, not separate unrelated cards:

```text
Source -> Metadata -> Preview -> Index
```

Source types:

- File upload.
- Public URL.
- Audio lecture.

The preview step displays:

- Detected title and type.
- Extracted character/page count.
- Metadata fields requiring confirmation.
- Chunk-size configuration.
- First chunks before indexing.

The execution state displays per-file progress, warnings, failures, and a final summary. Partial success must be visible.

Image ingestion stays hidden until OCR/caption is production-ready.

## Evaluation Workspace

### Configuration Strip

- Evaluation dataset.
- RAG mode.
- `top_k`.
- Run action.

### Results

- Compact metric row: Hit@K, MRR, citation rate, safety accuracy.
- Question-level results table.
- Comparison between `naive` and `hybrid`.
- Failure inspector containing expected source, retrieved sources, scores, and answer.

Metrics must include labels and definitions. Do not present isolated large numbers without context.

## Settings

- FastAPI base URL.
- Default corpus.
- Default RAG mode.
- Default `top_k`.
- Theme preference.
- Read-only display of active STT/LLM/embedding providers.

Secrets are configured on the backend and must never be entered or stored in the browser.

## Visual System

### Color Strategy

Restrained dark scientific palette using OKLCH tokens:

```text
--background:     oklch(0.16 0.018 245)
--surface-1:      oklch(0.19 0.018 245)
--surface-2:      oklch(0.22 0.019 245)
--border:         oklch(0.31 0.020 245)
--text-primary:   oklch(0.93 0.010 230)
--text-secondary: oklch(0.70 0.018 235)
--accent:         oklch(0.76 0.125 205)
--success:        oklch(0.72 0.115 155)
--warning:        oklch(0.80 0.115 85)
--danger:         oklch(0.68 0.155 25)
```

Accent is reserved for selected navigation, citation links, focus rings, and primary actions.

### Typography

- Inter or system sans.
- Body: 14-16px.
- Dense labels: 12-13px.
- Page headings: 20-24px.
- Answer line length: 68-72ch.
- Letter spacing: `0`.

### Geometry

- Radius scale: 4px, 6px, 8px.
- Control height: 32px compact, 40px standard.
- Icon size: 16px or 18px.
- Border: 1px.
- Focus ring: 2px accent with visible offset.

### Motion

- 150-220ms ease-out transitions.
- Animate opacity and transform only.
- Panel drawers and citation selection use motion to clarify state.
- Respect `prefers-reduced-motion`.

## Accessibility

- WCAG AA contrast.
- Complete keyboard navigation.
- Visible focus states.
- Semantic heading structure.
- Citation buttons have descriptive accessible labels.
- Source drawers trap focus and restore it on close.
- Status changes use appropriate live regions.
- Icons are never the only accessible label.

## Component Inventory

```text
AppShell
PrimaryNavigation
BackendStatus
ConversationHeader
MessageList
UserMessage
AssistantAnswer
CitationButton
SafetyNotice
ChatComposer
VoiceRecorder
EvidenceInspector
EvidenceSource
DocumentTable
DocumentFilters
DocumentDetail
IngestStepper
UploadQueue
ChunkPreview
EvaluationControls
MetricSummary
EvaluationResultsTable
FailureInspector
EmptyState
ErrorState
LoadingSkeleton
```

## Required Backend Contracts

Existing endpoints can support initial chat and uploads, but the full interface also requires:

```text
GET    /documents
GET    /documents/{doc_id}
GET    /documents/{doc_id}/chunks
DELETE /documents/{doc_id}
POST   /documents/{doc_id}/reindex

GET    /conversations
POST   /conversations
GET    /conversations/{id}
DELETE /conversations/{id}

POST   /ingest/preview
GET    /ingest/jobs/{job_id}

GET    /evaluation/datasets
POST   /evaluation/runs
GET    /evaluation/runs/{run_id}
```

The backend must also add configurable CORS origins for the standalone frontend.

## Acceptance Criteria

- A user can ask a question and inspect every citation without leaving chat.
- Only implemented RAG modes are visible.
- A user can locate and inspect indexed documents.
- Ingest provides preview and per-file outcome.
- Evaluation failures are inspectable at question and source level.
- Desktop, tablet, and mobile layouts are coherent.
- No medical answer is visually detached from its safety state and evidence.


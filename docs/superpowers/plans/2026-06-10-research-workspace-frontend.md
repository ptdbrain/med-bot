# MedTutor Research Workspace Frontend Implementation Plan

**Goal:** Build a standalone, production-quality frontend for MedTutor using the approved Research Workspace design.

**Architecture:** Create `web/` as a Next.js App Router TypeScript application. Use server-rendered layouts for the application shell, client components only for interactive workspaces, TanStack Query for FastAPI server state, and a small Zustand store for ephemeral panel/composer state.

**Frontend Stack:** Next.js App Router, TypeScript, Tailwind CSS, shadcn/ui component source, Lucide icons, TanStack Query, Zustand, React Markdown, Zod, Vitest, Testing Library, Playwright.

---

## Milestone 0: Backend Contracts

### Task 0.1: CORS

**Backend files:**

- Modify `app/config.py`
- Modify `app/main.py`
- Add tests for allowed and disallowed origins

Add `FRONTEND_ORIGINS` and FastAPI `CORSMiddleware`.

### Task 0.2: Document Queries

**Backend files:**

- Extend `app/retrieval/metadata_store.py`
- Add `app/api/documents.py`
- Add `app/schemas/document_api.py`

Implement list, detail, chunks, delete, and re-index contracts required by the document workspace.

### Task 0.3: Ingest Preview and Jobs

Split extraction/chunk preview from index mutation. Add a minimal persisted job model for progress and partial failures.

### Task 0.4: Evaluation Details

Return per-question evaluation results instead of the current empty `details` list. Add stored evaluation runs only after the synchronous version is reliable.

The first release stores conversation history in the browser. Backend conversation persistence is a separate follow-up contract after the chat interaction is validated.

---

## Milestone 1: Frontend Foundation

### Task 1.1: Scaffold

Create:

```text
web/
├── app/
├── components/
├── features/
├── lib/
├── stores/
├── tests/
└── public/
```

Initialize Next.js App Router with TypeScript, Tailwind, ESLint, and the `@/*` alias.

### Task 1.2: Design Tokens

Create semantic CSS variables from the approved OKLCH palette. Configure dark mode, focus rings, radius scale, typography, and reduced motion.

### Task 1.3: Providers

Add:

- TanStack Query provider.
- Theme provider.
- Global error boundary.
- Toast region.
- API base URL validation.

### Task 1.4: API Client

Create typed modules:

```text
lib/api/client.ts
lib/api/chat.ts
lib/api/documents.ts
lib/api/ingest.ts
lib/api/evaluation.ts
lib/api/voice.ts
lib/api/health.ts
```

Normalize FastAPI errors into one frontend error type.

---

## Milestone 2: App Shell

### Task 2.1: Navigation

Implement desktop sidebar, tablet icon rail, and mobile navigation sheet.

### Task 2.2: Status

Poll `/health` and display:

- Online/offline.
- Indexed chunks.
- Active corpus when the backend exposes a corpus identifier; omit this row before that contract exists.

### Task 2.3: Route Skeletons

Create usable empty routes for Chat, Documents, Ingest, Evaluation, and Settings. Add loading and error boundaries per route.

---

## Milestone 3: Chat and Evidence

### Task 3.1: Chat Types and Local History

Define conversation/message/source types. Persist conversation summaries locally until backend persistence is added.

### Task 3.2: Composer

Implement keyboard behavior, submit state, attachment affordance, microphone entry point, and accessible errors.

### Task 3.3: Answer Rendering

Render Markdown safely. Convert citation markers into interactive `CitationButton` elements linked to returned sources.

### Task 3.4: Evidence Inspector

Implement synchronized citation selection, excerpt display, source metadata, score formatting, and mobile drawer behavior.

### Task 3.5: Safety States

Add dedicated visual treatments for emergency, blocked medical advice, insufficient context, and backend errors.

### Task 3.6: Tests

Test:

- Submit and loading behavior.
- Citation-to-source synchronization.
- Empty context.
- Emergency response.
- Responsive evidence drawer.

---

## Milestone 4: Documents

### Task 4.1: Table and Filters

Implement searchable document table with URL-driven filters and sorting.

### Task 4.2: Document Detail

Render metadata, sections, chunks, original source, index status, and re-index action.

### Task 4.3: Empty and Error States

Handle no documents, unavailable source, failed index, and backend offline.

---

## Milestone 5: Ingest

### Task 5.1: Source Selection

Support file, URL, and audio lecture.

### Task 5.2: Metadata and Preview

Display extracted metadata and chunk preview before index mutation.

### Task 5.3: Execution

Show per-file progress, warnings, errors, and final results. Preserve successful items when another item fails.

### Task 5.4: Upload Hardening

Validate extension and size client-side while treating backend validation as authoritative.

---

## Milestone 6: Evaluation

### Task 6.1: Controls

Dataset, RAG mode, and `top_k`.

### Task 6.2: Results

Metric summary, question table, naive/hybrid comparison, and failure inspector.

### Task 6.3: Interpretation

Include metric definitions and source-level debugging information. Avoid unexplained scores.

---

## Milestone 7: Audio

### Task 7.1: Audio Upload

Connect transcribe, voice chat, and lecture ingest endpoints.

### Task 7.2: Recorder

Add browser recording only after uploaded-audio flows pass cross-browser tests.

### Task 7.3: Transcript Review

Allow the user to review a transcript before sending it as a question or indexing it as a lecture.

---

## Milestone 8: Quality Gate

### Responsive

Verify at:

- 1440x900
- 1024x768
- 768x1024
- 390x844

### Accessibility

- Keyboard-only navigation.
- Screen-reader labels.
- Focus restoration for drawers.
- WCAG AA contrast.
- Reduced motion.

### Testing

```text
unit:       Vitest
components: Testing Library
e2e:        Playwright
API:        FastAPI contract tests
```

### Performance

- Lazy-load evidence excerpts and document chunk lists.
- Virtualize long document/evaluation tables only when measured.
- Keep initial Chat route free of document-management code.

---

## Delivery Order

1. Backend CORS and document-read APIs.
2. Frontend foundation and app shell.
3. Chat plus evidence inspector.
4. Documents.
5. Ingest preview/jobs.
6. Evaluation details.
7. Audio.
8. Accessibility, responsive and Playwright verification.

The first demonstrable release ends after step 3. It must already support a real question, grounded answer, citations, safety states, and mobile evidence inspection.

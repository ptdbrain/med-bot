import { apiRequest } from "@/lib/api/client";

export interface IngestResult {
  doc_id: string;
  title: string;
  chunks_added: number;
  source_path: string;
  source_type: string;
  transcription?: {
    text: string;
    provider: string;
    language: string | null;
    duration_seconds: number | null;
  };
}

export type MediaKind = "image" | "audio" | "video";

export interface IngestJob {
  job_id: string;
  filename: string;
  status: "queued" | "extracting" | "chunking" | "embedding" | "publishing" | "completed" | "needs_review" | "failed" | "cancelled";
  report: Partial<IngestResult> & { page_count?: number; usable_pages?: number; failed_pages?: number; warning_pages?: number };
  error: string | null;
}

export function createIngestJob(file: File) {
  const body = new FormData();
  body.append("file", file);
  return apiRequest<IngestJob>("/ingest/jobs", { method: "POST", body });
}

export function getIngestJob(id: string) {
  return apiRequest<IngestJob>(`/ingest/jobs/${encodeURIComponent(id)}`);
}

export function cancelIngestJob(id: string) {
  return apiRequest<IngestJob>(`/ingest/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
}

export function retryIngestJob(id: string) {
  return apiRequest<IngestJob>(`/ingest/jobs/${encodeURIComponent(id)}/retry`, { method: "POST" });
}

function upload(path: string, file: File) {
  const body = new FormData();
  body.append("file", file);
  return apiRequest<IngestResult>(path, {
    method: "POST",
    body,
  });
}

export function uploadDocument(file: File) {
  return upload("/ingest/file", file);
}

export function uploadMedia(file: File, kind: MediaKind) {
  const paths: Record<MediaKind, string> = {
    image: "/image/ingest",
    audio: "/voice/ingest",
    video: "/video/ingest",
  };
  return upload(paths[kind], file);
}

export function ingestUrl(url: string) {
  return apiRequest<IngestResult>("/ingest/url", {
    method: "POST",
    body: JSON.stringify({ url }),
  });
}

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

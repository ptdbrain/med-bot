import { apiRequest } from "@/lib/api/client";

export interface DocumentRecord {
  doc_id: string;
  title: string;
  source_path: string;
  source_type: string;
  chunks_count: number;
  metadata: Record<string, unknown>;
}

export interface DocumentsResponse {
  documents: DocumentRecord[];
  total: number;
}

export function getDocuments() {
  return apiRequest<DocumentsResponse>("/documents");
}

export function deleteDocument(docId: string) {
  return apiRequest<{ doc_id: string; deleted: boolean }>(
    `/documents/${encodeURIComponent(docId)}`,
    { method: "DELETE" },
  );
}

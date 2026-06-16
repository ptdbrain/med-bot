import { apiRequest } from "@/lib/api/client";

export type RagMode =
  | "naive"
  | "hybrid"
  | "rerank"
  | "parent_child"
  | "multi_query"
  | "corrective"
  | "agentic";

export interface ChatSource {
  chunk_id: string;
  title: string;
  source_path: string;
  source_type: string;
  score: number;
  page_number: number | null;
  section_title: string | null;
  excerpt?: string;
}

interface RetrievedChunk {
  chunk: {
    chunk_id: string;
    text: string;
  };
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  retrieved_chunks: RetrievedChunk[];
  mode_used: RagMode;
  safety_warning: "emergency" | "blocked_medical_advice" | null;
}

export interface ChatRequest {
  question: string;
  rag_mode: RagMode;
  top_k: number;
}

export function enrichSources(response: ChatResponse): ChatResponse {
  const excerpts = new Map(
    response.retrieved_chunks.map((item) => [
      item.chunk.chunk_id,
      item.chunk.text,
    ]),
  );

  return {
    ...response,
    sources: response.sources.map((source) => ({
      ...source,
      excerpt: excerpts.get(source.chunk_id),
    })),
  };
}

export async function sendChat(payload: ChatRequest) {
  const response = await apiRequest<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return enrichSources(response);
}

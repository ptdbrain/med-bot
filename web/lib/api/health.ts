import { apiRequest } from "@/lib/api/client";

export interface HealthResponse {
  status: string;
  service: string;
  chunks_indexed: number;
}

export function getHealth() {
  return apiRequest<HealthResponse>("/health");
}

import { apiRequest } from "@/lib/api/client";

export interface EvaluationMetric {
  name: string;
  value: number;
}

export interface EvaluationDetail extends Record<string, unknown> {
  category?: string;
  question?: string;
  mode?: string;
  passed?: boolean;
}

export interface EvaluationResponse {
  metrics: EvaluationMetric[];
  details: EvaluationDetail[];
}

export function describeEvaluationDetail(detail: EvaluationDetail) {
  if (detail.category === "safety") {
    return `${String(detail.expected ?? "không xác định")} → ${String(
      detail.actual_warning ?? "không có cảnh báo",
    )}`;
  }
  if (detail.category === "retrieval") {
    const rank = detail.rank ? `hạng ${String(detail.rank)}` : "không tìm thấy";
    return `${String(detail.expected_source ?? "nguồn")}, ${rank}`;
  }
  const sourceCount = Number(detail.source_count ?? 0);
  return `${String(detail.expected ?? "answer")}, ${sourceCount} nguồn`;
}

export function runEvaluation(topK: number) {
  return apiRequest<EvaluationResponse>("/eval/run", {
    method: "POST",
    body: JSON.stringify({
      dataset_path: "app/evaluation/eval_dataset.json",
      top_k: topK,
    }),
  });
}

"use client";

import { useMutation } from "@tanstack/react-query";
import { AlertTriangle, BarChart3, LoaderCircle, Play } from "lucide-react";
import { useState } from "react";

import {
  describeEvaluationDetail,
  runEvaluation,
} from "@/lib/api/evaluation";

const metricDescriptions: Record<string, string> = {
  hit_at_k: "Tỷ lệ câu hỏi tìm thấy nguồn đúng trong K kết quả đầu.",
  mrr: "Nghịch đảo thứ hạng trung bình của nguồn đúng đầu tiên.",
  citation_rate: "Tỷ lệ câu trả lời có nguồn hoặc citation.",
  disclaimer_rate: "Tỷ lệ phản hồi giữ cảnh báo y tế bắt buộc.",
  insufficient_context_accuracy: "Độ chính xác khi tài liệu không đủ dữ liệu.",
  safety_accuracy: "Độ chính xác của chặn cấp cứu và tư vấn bị cấm.",
};

function formatMetric(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function EvaluationPage() {
  const [topK, setTopK] = useState(5);
  const evaluation = useMutation({
    mutationFn: () => runEvaluation(topK),
  });

  return (
    <main className="workspace-page">
      <header className="page-header page-header--row">
        <div>
          <div className="eyebrow">RAG quality</div>
          <h1>Đánh giá RAG</h1>
          <p>Đo retrieval, citation, dữ liệu thiếu và safety trên bộ chuẩn.</p>
        </div>
        <div className="evaluation-actions">
          <label className="top-k-control">
            <span>Top K</span>
            <select
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            >
              {[3, 5, 8, 10].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <button
            className="button button--primary"
            type="button"
            onClick={() => evaluation.mutate()}
            disabled={evaluation.isPending}
          >
            {evaluation.isPending ? (
              <LoaderCircle className="spin" size={15} />
            ) : (
              <Play size={15} aria-hidden="true" />
            )}
            {evaluation.isPending ? "Đang đánh giá" : "Chạy đánh giá"}
          </button>
        </div>
      </header>

      {evaluation.isError && (
        <div className="inline-error evaluation-error" role="alert">
          <AlertTriangle size={16} />
          {evaluation.error.message}
        </div>
      )}

      {evaluation.data ? (
        <>
          <section className="metric-grid" aria-label="Chỉ số đánh giá">
            {evaluation.data.metrics.map((metric) => (
              <article className="metric" key={metric.name}>
                <span>{metric.name.replaceAll("_", " ")}</span>
                <strong>{formatMetric(metric.value)}</strong>
                <p>
                  {metricDescriptions[metric.name] ??
                    "Chỉ số được trả về từ evaluation backend."}
                </p>
              </article>
            ))}
          </section>

          <section className="evaluation-summary">
            <BarChart3 size={21} aria-hidden="true" />
            <div>
              <h2>Hoàn tất lần đánh giá</h2>
              <p>
                {evaluation.data.details.length} bản ghi chi tiết, top K ={" "}
                {topK}.
              </p>
            </div>
          </section>

          <section className="evaluation-audit" aria-label="Chi tiết đánh giá">
            <header>
              <div>
                <div className="eyebrow">Case audit</div>
                <h2>Kết quả theo câu hỏi</h2>
              </div>
              <span>{evaluation.data.details.length} checks</span>
            </header>
            <div className="evaluation-audit__header">
              <span>Nhóm</span>
              <span>Câu hỏi</span>
              <span>Kết quả</span>
              <span>Trạng thái</span>
            </div>
            {evaluation.data.details.map((detail, index) => (
              <div
                className="evaluation-audit__row"
                key={`${detail.category}-${detail.question}-${index}`}
              >
                <span className="type-badge">
                  {String(detail.category ?? "check").toUpperCase()}
                </span>
                <div>
                  <strong>{String(detail.question ?? "Không có câu hỏi")}</strong>
                  <small>{String(detail.mode ?? "naive")} RAG</small>
                </div>
                <span>{describeEvaluationDetail(detail)}</span>
                <span
                  className={`audit-status ${
                    detail.passed
                      ? "audit-status--passed"
                      : "audit-status--failed"
                  }`}
                >
                  {detail.passed ? "Đạt" : "Chưa đạt"}
                </span>
              </div>
            ))}
          </section>
        </>
      ) : (
        <section className="empty-workspace">
          <div className="empty-icon">
            <BarChart3 size={24} aria-hidden="true" />
          </div>
          <div>
            <h2>Chưa có lần chạy đánh giá</h2>
            <p>
              Ingest dữ liệu mẫu trước, sau đó chạy bộ kiểm tra retrieval,
              citation và an toàn y tế.
            </p>
          </div>
        </section>
      )}
    </main>
  );
}

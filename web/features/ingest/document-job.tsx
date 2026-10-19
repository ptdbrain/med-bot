"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileUp, LoaderCircle, RotateCcw, X } from "lucide-react";
import { useEffect, useState } from "react";
import { cancelIngestJob, createIngestJob, getIngestJob, retryIngestJob } from "@/lib/api/ingest";

const terminal = new Set(["completed", "needs_review", "failed", "cancelled"]);
const labels: Record<string, string> = {
  queued: "Đang chờ worker", extracting: "Đang đọc tài liệu / OCR", chunking: "Đang chia đoạn",
  embedding: "Đang lập chỉ mục", publishing: "Đang công bố phiên bản", completed: "Đã hoàn tất",
  needs_review: "Có trang cần kiểm tra", failed: "Xử lý thất bại", cancelled: "Đã hủy",
};

export function DocumentJob({ file }: { file: File | null }) {
  const client = useQueryClient();
  const [id, setId] = useState<string | null>(null);
  const upload = useMutation({ mutationFn: createIngestJob, onSuccess: (job) => setId(job.job_id) });
  const job = useQuery({ queryKey: ["ingest-job", id], queryFn: () => getIngestJob(id!), enabled: !!id,
    refetchInterval: (query) => terminal.has(query.state.data?.status ?? "") ? false : 1500 });
  const action = useMutation({ mutationFn: async (kind: "cancel" | "retry") => {
    if (!id) return;
    await (kind === "cancel" ? cancelIngestJob(id) : retryIngestJob(id));
    await client.invalidateQueries({ queryKey: ["ingest-job", id] });
  }});
  const status = job.data?.status;
  const busy = upload.isPending || (!!id && !terminal.has(status ?? ""));
  useEffect(() => {
    if (status === "completed") {
      void client.invalidateQueries({ queryKey: ["documents"] });
      void client.invalidateQueries({ queryKey: ["health"] });
    }
  }, [status, client]);
  const error = upload.error ?? job.error ?? action.error;
  return <div className="ingest-feedback">
    <button className="button button--primary" type="button" disabled={!file || busy} onClick={() => file && upload.mutate(file)}>
      {busy ? <LoaderCircle className="spin" size={16} /> : <FileUp size={16} />}
      {busy ? "Đang xử lý" : "Ingest tài liệu"}
    </button>
    {job.data && <div role="status" aria-live="polite">
      <p><strong>{job.data.filename}</strong>: {labels[job.data.status]}</p>
      {job.data.report.page_count !== undefined && <p>{job.data.report.usable_pages ?? 0}/{job.data.report.page_count} trang đọc được; {job.data.report.warning_pages ?? 0} trang có cảnh báo.</p>}
      {job.data.report.chunks_added !== undefined && <p>{job.data.report.chunks_added} đoạn đã được lập chỉ mục.</p>}
      {status === "needs_review" && <p>Tài liệu chưa được công bố. Hãy kiểm tra chất lượng bản scan hoặc cấu hình OCR rồi thử lại.</p>}
      {status === "queued" && <p>Chưa bắt đầu xử lý. Kiểm tra dịch vụ ingest worker nếu trạng thái này kéo dài.</p>}
      {busy && <button type="button" className="button" disabled={action.isPending} onClick={() => action.mutate("cancel")}><X size={16} /> Hủy</button>}
      {status && ["failed", "needs_review", "cancelled"].includes(status) && <button type="button" className="button" disabled={action.isPending} onClick={() => action.mutate("retry")}><RotateCcw size={16} /> Thử lại</button>}
      {job.data.error && <p role="alert">{job.data.error}</p>}
    </div>}
    {error && <p className="inline-error" role="alert">{error.message}</p>}
  </div>;
}

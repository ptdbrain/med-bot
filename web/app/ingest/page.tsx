"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  FileText,
  FileUp,
  ImageIcon,
  Link2,
  LoaderCircle,
  Mic2,
  Video,
} from "lucide-react";
import { FormEvent, useState } from "react";
import { DocumentJob } from "@/features/ingest/document-job";

import {
  ingestUrl,
  type IngestResult,
  uploadDocument,
  uploadMedia,
} from "@/lib/api/ingest";

type SourceKind = "document" | "url" | "image" | "audio" | "video";

const sourceTypes: Array<{
  kind: SourceKind;
  icon: typeof FileUp;
  title: string;
  description: string;
  accept?: string;
}> = [
  {
    kind: "document",
    icon: FileUp,
    title: "Tệp tài liệu",
    description: "PDF, DOCX, Markdown, HTML hoặc văn bản.",
    accept: ".txt,.md,.pdf,.docx,.html,.htm",
  },
  {
    kind: "url",
    icon: Link2,
    title: "URL công khai",
    description: "Thu thập một trang HTML dùng cho học tập.",
  },
  {
    kind: "image",
    icon: ImageIcon,
    title: "Ảnh học tập",
    description: "OCR và caption ảnh PNG hoặc JPEG.",
    accept: ".png,.jpg,.jpeg",
  },
  {
    kind: "audio",
    icon: Mic2,
    title: "Bài giảng âm thanh",
    description: "Chuyển transcript thành tài liệu có thể tìm kiếm.",
    accept: ".wav,.mp3,.m4a,.ogg",
  },
  {
    kind: "video",
    icon: Video,
    title: "Video bài giảng",
    description: "Trích transcript để đưa nội dung video vào RAG.",
    accept: ".mp4,.mov,.mkv,.webm",
  },
];

export default function IngestPage() {
  const queryClient = useQueryClient();
  const [kind, setKind] = useState<SourceKind>("document");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");

  const mutation = useMutation({
    mutationFn: async (): Promise<IngestResult> => {
      if (kind === "url") return ingestUrl(url.trim());
      if (!file) throw new Error("Hãy chọn một tệp trước khi ingest.");
      if (kind === "document") return uploadDocument(file);
      return uploadMedia(file, kind);
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["documents"] }),
        queryClient.invalidateQueries({ queryKey: ["health"] }),
      ]);
    },
  });

  const selected = sourceTypes.find((source) => source.kind === kind)!;

  function selectKind(nextKind: SourceKind) {
    setKind(nextKind);
    setFile(null);
    setUrl("");
    mutation.reset();
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (kind === "document") return;
    if (kind === "url" && !url.trim()) return;
    if (kind !== "url" && !file) return;
    mutation.mutate();
  }

  return (
    <main className="workspace-page">
      <header className="page-header">
        <div>
          <div className="eyebrow">Data pipeline</div>
          <h1>Ingest dữ liệu</h1>
          <p>Đưa nguồn học tập vào kho, kèm metadata và chỉ mục truy hồi.</p>
        </div>
      </header>

      <ol className="stepper" aria-label="Quy trình ingest">
        {["Nguồn", "Tệp hoặc URL", "Xử lý", "Lập chỉ mục"].map(
          (step, index) => (
            <li
              key={step}
              className={
                mutation.isSuccess || index === 0
                  ? "stepper__item--active"
                  : ""
              }
            >
              <span>{index + 1}</span>
              {step}
            </li>
          ),
        )}
      </ol>

      <section className="source-picker" aria-labelledby="source-heading">
        <div className="section-heading">
          <div className="eyebrow">Chọn nguồn</div>
          <h2 id="source-heading">Dữ liệu cần lập chỉ mục</h2>
        </div>
        <div className="source-options">
          {sourceTypes.map(({ kind: sourceKind, icon: Icon, title, description }) => (
            <button
              key={sourceKind}
              className={`source-option ${
                kind === sourceKind ? "source-option--active" : ""
              }`}
              type="button"
              onClick={() => selectKind(sourceKind)}
              aria-pressed={kind === sourceKind}
            >
              <Icon size={20} aria-hidden="true" />
              <span>
                <strong>{title}</strong>
                <small>{description}</small>
              </span>
            </button>
          ))}
        </div>

        <form className="ingest-form" onSubmit={handleSubmit}>
          {kind === "url" ? (
            <label className="field">
              URL công khai
              <input
                type="url"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://example.org/guideline"
                required
              />
            </label>
          ) : (
            <label className="file-drop">
              <selected.icon size={22} aria-hidden="true" />
              <span>
                <strong>{file?.name ?? selected.title}</strong>
                <small>
                  {file
                    ? `${(file.size / 1024).toFixed(1)} KB`
                    : `Chọn tệp tối đa 15 MB (${selected.accept})`}
                </small>
              </span>
              <input
                type="file"
                accept={selected.accept}
                onChange={(event) =>
                  setFile(event.target.files?.[0] ?? null)
                }
                required
              />
            </label>
          )}

          {kind === "document" ? <DocumentJob file={file} /> : <button
            className="button button--primary"
            type="submit"
            disabled={
              mutation.isPending ||
              (kind === "url" ? !url.trim() : !file)
            }
          >
            {mutation.isPending ? (
              <LoaderCircle className="spin" size={16} aria-hidden="true" />
            ) : (
              <FileText size={16} aria-hidden="true" />
            )}
            {mutation.isPending ? "Đang xử lý" : "Ingest và lập chỉ mục"}
          </button>}
        </form>

        {mutation.isError && (
          <div className="inline-error ingest-feedback" role="alert">
            {mutation.error.message}
          </div>
        )}

        {mutation.isSuccess && (
          <div className="ingest-success" aria-live="polite">
            <CheckCircle2 size={19} aria-hidden="true" />
            <div>
              <strong>{mutation.data.title}</strong>
              <span>
                {mutation.data.chunks_added} chunks,{" "}
                {mutation.data.source_type.toUpperCase()}
              </span>
              <code>{mutation.data.doc_id}</code>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

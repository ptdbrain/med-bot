"use client";

import {
  ArrowLeft,
  ArrowRight,
  ExternalLink,
  FileText,
  PanelRightClose,
  X,
} from "lucide-react";

import type { ChatSource } from "@/lib/api/chat";

function sourceLocation(source: ChatSource) {
  const parts = [
    source.page_number ? `Trang ${source.page_number}` : null,
    source.section_title,
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "Không có thông tin vị trí";
}

export function EvidenceInspector({
  sources,
  activeIndex,
  onSelect,
  onClose,
  compactOpen = false,
}: {
  sources: ChatSource[];
  activeIndex: number;
  onSelect: (index: number) => void;
  onClose: () => void;
  compactOpen?: boolean;
}) {
  const safeIndex = Math.min(Math.max(activeIndex, 0), sources.length - 1);
  const source = sources[safeIndex];
  const canOpen =
    source?.source_path.startsWith("http://") ||
    source?.source_path.startsWith("https://");

  return (
    <aside
      className={`evidence-panel ${
        compactOpen ? "evidence-panel--compact-open" : ""
      }`}
      aria-label="Bằng chứng trích dẫn"
    >
      <header className="evidence-header">
        <div>
          <div className="eyebrow">Evidence</div>
          <h2>Bằng chứng truy xuất</h2>
        </div>
        <button
          className="icon-button desktop-close"
          type="button"
          onClick={onClose}
          aria-label="Đóng panel bằng chứng"
          title="Đóng panel"
        >
          <PanelRightClose size={18} />
        </button>
        <button
          className="icon-button mobile-close"
          type="button"
          onClick={onClose}
          aria-label="Đóng bằng chứng"
        >
          <X size={18} />
        </button>
      </header>

      {!source ? (
        <div className="evidence-empty">
          <FileText size={23} aria-hidden="true" />
          <h3>Chưa có nguồn được chọn</h3>
          <p>
            Sau khi có câu trả lời, chọn citation trong nội dung để xem chunk
            bằng chứng tương ứng.
          </p>
        </div>
      ) : (
        <div className="evidence-body">
          <div className="source-position">
            <span>
              Nguồn {safeIndex + 1}/{sources.length}
            </span>
            <div>
              <button
                className="icon-button"
                type="button"
                disabled={safeIndex === 0}
                onClick={() => onSelect(safeIndex - 1)}
                aria-label="Nguồn trước"
              >
                <ArrowLeft size={16} />
              </button>
              <button
                className="icon-button"
                type="button"
                disabled={safeIndex === sources.length - 1}
                onClick={() => onSelect(safeIndex + 1)}
                aria-label="Nguồn tiếp theo"
              >
                <ArrowRight size={16} />
              </button>
            </div>
          </div>

          <div className="source-identity">
            <span className="source-number">{safeIndex + 1}</span>
            <div>
              <h3>{source.title}</h3>
              <p>{sourceLocation(source)}</p>
            </div>
          </div>

          <dl className="source-metadata">
            <div>
              <dt>Loại</dt>
              <dd>{source.source_type.toUpperCase()}</dd>
            </div>
            <div>
              <dt>Điểm retrieval</dt>
              <dd>{source.score.toFixed(3)}</dd>
            </div>
            <div>
              <dt>Chunk ID</dt>
              <dd title={source.chunk_id}>{source.chunk_id}</dd>
            </div>
          </dl>

          <section className="excerpt-section">
            <div className="excerpt-heading">Đoạn được truy xuất</div>
            <blockquote>
              {source.excerpt ?? "Backend không trả về nội dung chunk này."}
            </blockquote>
          </section>

          {canOpen && (
            <a
              className="button evidence-open"
              href={source.source_path}
              target="_blank"
              rel="noreferrer"
            >
              <ExternalLink size={15} aria-hidden="true" />
              Mở tài liệu gốc
            </a>
          )}
        </div>
      )}
    </aside>
  );
}

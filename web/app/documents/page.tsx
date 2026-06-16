"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  FileStack,
  LoaderCircle,
  Search,
  Trash2,
} from "lucide-react";
import { useMemo, useState } from "react";

import {
  deleteDocument,
  getDocuments,
  type DocumentRecord,
} from "@/lib/api/documents";

function DocumentRow({
  document,
  deleting,
  confirmDelete,
  onDelete,
  onCancel,
}: {
  document: DocumentRecord;
  deleting: boolean;
  confirmDelete: boolean;
  onDelete: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="data-row">
      <div className="document-name">
        <FileStack size={17} aria-hidden="true" />
        <span>
          <strong>{document.title}</strong>
          <small>{document.source_path}</small>
        </span>
      </div>
      <span className="type-badge">{document.source_type.toUpperCase()}</span>
      <span>{document.chunks_count}</span>
      <div className="row-actions">
        {confirmDelete ? (
          <>
            <button
              className="button button--danger"
              type="button"
              onClick={onDelete}
              disabled={deleting}
            >
              {deleting ? (
                <LoaderCircle className="spin" size={14} />
              ) : (
                "Xóa"
              )}
            </button>
            <button className="button" type="button" onClick={onCancel}>
              Hủy
            </button>
          </>
        ) : (
          <button
            className="icon-button"
            type="button"
            onClick={onDelete}
            aria-label={`Xóa ${document.title}`}
            title="Xóa tài liệu"
          >
            <Trash2 size={16} />
          </button>
        )}
      </div>
    </div>
  );
}

export default function DocumentsPage() {
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const documents = useQuery({
    queryKey: ["documents"],
    queryFn: getDocuments,
  });
  const deletion = useMutation({
    mutationFn: deleteDocument,
    onSuccess: async () => {
      setPendingDelete(null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["documents"] }),
        queryClient.invalidateQueries({ queryKey: ["health"] }),
      ]);
    },
  });

  const filtered = useMemo(() => {
    const term = query.trim().toLocaleLowerCase("vi");
    const rows = documents.data?.documents ?? [];
    if (!term) return rows;
    return rows.filter((document) =>
      [document.title, document.source_path, document.source_type]
        .join(" ")
        .toLocaleLowerCase("vi")
        .includes(term),
    );
  }, [documents.data, query]);

  return (
    <main className="workspace-page">
      <header className="page-header page-header--row">
        <div>
          <div className="eyebrow">Nguồn tri thức</div>
          <h1>Kho tài liệu</h1>
          <p>{documents.data?.total ?? 0} tài liệu được phép dùng làm bằng chứng.</p>
        </div>
        <a className="button button--primary" href="/ingest">
          Thêm tài liệu
        </a>
      </header>

      <div className="table-toolbar">
        <label className="search-field">
          <Search size={16} aria-hidden="true" />
          <span className="sr-only">Tìm tài liệu</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Tìm theo tiêu đề, đường dẫn hoặc loại"
          />
        </label>
      </div>

      {deletion.isError && (
        <div className="inline-error table-error" role="alert">
          <AlertTriangle size={16} />
          {deletion.error.message}
        </div>
      )}

      <section className="data-region" aria-label="Danh sách tài liệu">
        <div className="data-header">
          <span>Tài liệu</span>
          <span>Loại</span>
          <span>Chunks</span>
          <span>Thao tác</span>
        </div>

        {documents.isPending ? (
          <div className="table-loading" aria-live="polite">
            <LoaderCircle className="spin" size={20} />
            Đang tải kho tài liệu
          </div>
        ) : documents.isError ? (
          <div className="empty-table">
            <AlertTriangle size={24} aria-hidden="true" />
            <h2>Không tải được kho tài liệu</h2>
            <p>{documents.error.message}</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-table">
            <FileStack size={24} aria-hidden="true" />
            <h2>{query ? "Không có kết quả phù hợp" : "Kho tài liệu đang trống"}</h2>
            <p>
              {query
                ? "Thử từ khóa khác hoặc xóa bộ lọc hiện tại."
                : "Ingest tài liệu học tập để bắt đầu truy hồi có dẫn nguồn."}
            </p>
          </div>
        ) : (
          filtered.map((document) => (
            <DocumentRow
              key={document.doc_id}
              document={document}
              confirmDelete={pendingDelete === document.doc_id}
              deleting={
                deletion.isPending && deletion.variables === document.doc_id
              }
              onDelete={() => {
                if (pendingDelete === document.doc_id) {
                  deletion.mutate(document.doc_id);
                } else {
                  setPendingDelete(document.doc_id);
                }
              }}
              onCancel={() => setPendingDelete(null)}
            />
          ))
        )}
      </section>
    </main>
  );
}

"use client";

import { useQuery } from "@tanstack/react-query";
import { Database } from "lucide-react";

import { getHealth } from "@/lib/api/health";

export function BackendStatus({ compact = false }: { compact?: boolean }) {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 15_000,
  });

  const online = health.isSuccess;
  const label = online
    ? `${health.data.chunks_indexed} chunks`
    : health.isPending
      ? "Đang kết nối"
      : "Backend offline";

  return (
    <div
      className={`backend-status ${compact ? "backend-status--compact" : ""}`}
      aria-live="polite"
      title={label}
    >
      <span
        className={`status-dot ${online ? "status-dot--online" : "status-dot--offline"}`}
      />
      {!compact && (
        <>
          <Database aria-hidden="true" size={15} />
          <span>{label}</span>
        </>
      )}
    </div>
  );
}

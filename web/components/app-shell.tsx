"use client";

import {
  BarChart3,
  Database,
  FileStack,
  Menu,
  MessageSquareText,
  Microscope,
  Settings,
  Upload,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { BackendStatus } from "@/components/backend-status";

const navigation = [
  { href: "/chat", label: "Chat", icon: MessageSquareText },
  { href: "/documents", label: "Kho tài liệu", icon: FileStack },
  { href: "/ingest", label: "Ingest dữ liệu", icon: Upload },
  { href: "/evaluation", label: "Đánh giá RAG", icon: BarChart3 },
  { href: "/settings", label: "Cài đặt", icon: Settings },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="app-frame">
      <aside
        className={`primary-nav ${mobileOpen ? "primary-nav--open" : ""}`}
        aria-label="Điều hướng chính"
      >
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true">
            <Microscope size={19} />
          </div>
          <div className="brand-copy">
            <strong>MedTutor</strong>
            <span>Research Workspace</span>
          </div>
          <button
            className="icon-button nav-close"
            type="button"
            onClick={() => setMobileOpen(false)}
            aria-label="Đóng điều hướng"
          >
            <X size={18} />
          </button>
        </div>

        <nav className="nav-links">
          {navigation.map((item) => {
            const Icon = item.icon;
            const active = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileOpen(false)}
                className={`nav-link ${active ? "nav-link--active" : ""}`}
                aria-current={active ? "page" : undefined}
                title={item.label}
              >
                <Icon size={18} aria-hidden="true" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="nav-context">
          <div className="context-label">Corpus hiện tại</div>
          <div className="context-value">
            <Database size={15} aria-hidden="true" />
            <span>MedTutor Library</span>
          </div>
          <BackendStatus />
        </div>
      </aside>

      {mobileOpen && (
        <button
          className="nav-scrim"
          type="button"
          aria-label="Đóng điều hướng"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <div className="app-main">
        <header className="mobile-bar">
          <button
            className="icon-button"
            type="button"
            onClick={() => setMobileOpen(true)}
            aria-label="Mở điều hướng"
          >
            <Menu size={19} />
          </button>
          <div className="mobile-brand">
            <Microscope size={18} aria-hidden="true" />
            <span>MedTutor</span>
          </div>
          <BackendStatus compact />
        </header>
        {children}
      </div>
    </div>
  );
}

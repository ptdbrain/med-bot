"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { RagMode } from "@/lib/api/chat";

interface WorkspaceState {
  ragMode: RagMode;
  topK: number;
  evidenceOpen: boolean;
  activeSourceIndex: number;
  setRagMode: (mode: RagMode) => void;
  setTopK: (topK: number) => void;
  setEvidenceOpen: (open: boolean) => void;
  selectSource: (index: number) => void;
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      ragMode: "hybrid",
      topK: 5,
      evidenceOpen: true,
      activeSourceIndex: 0,
      setRagMode: (ragMode) => set({ ragMode }),
      setTopK: (topK) => set({ topK }),
      setEvidenceOpen: (evidenceOpen) => set({ evidenceOpen }),
      selectSource: (activeSourceIndex) =>
        set({ activeSourceIndex, evidenceOpen: true }),
    }),
    {
      name: "medtutor-workspace",
      partialize: ({ ragMode, topK }) => ({ ragMode, topK }),
    },
  ),
);

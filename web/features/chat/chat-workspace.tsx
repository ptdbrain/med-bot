"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  BookOpenCheck,
  Copy,
  FilePlus2,
  LoaderCircle,
  Mic,
  PanelRight,
  RotateCcw,
  Send,
  Settings2,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useRouter } from "next/navigation";

import { AnswerContent } from "@/features/chat/answer-content";
import { EvidenceInspector } from "@/features/chat/evidence-inspector";
import {
  sendChat,
  type ChatSource,
  type RagMode,
} from "@/lib/api/chat";
import { getHealth } from "@/lib/api/health";
import { useMediaQuery } from "@/lib/use-media-query";
import { useWorkspaceStore } from "@/stores/workspace-store";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: ChatSource[];
  mode?: RagMode;
  safetyWarning?: "emergency" | "blocked_medical_advice" | null;
}

const HISTORY_KEY = "medtutor-chat-history";
const examples = [
  "FAST trong đột quỵ gồm những dấu hiệu nào?",
  "Tóm tắt cơ chế tác dụng được mô tả trong tài liệu.",
  "So sánh hai khuyến cáo trong các nguồn đã ingest.",
];
const ragModes: Array<{ value: RagMode; label: string }> = [
  { value: "naive", label: "Naive" },
  { value: "hybrid", label: "Hybrid" },
  { value: "rerank", label: "Rerank" },
  { value: "parent_child", label: "Parent-child" },
  { value: "multi_query", label: "Multi-query" },
  { value: "corrective", label: "Corrective" },
  { value: "agentic", label: "Agentic" },
];

function readHistory(): ChatMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const stored = window.localStorage.getItem(HISTORY_KEY);
    return stored ? (JSON.parse(stored) as ChatMessage[]) : [];
  } catch {
    return [];
  }
}

export function ChatWorkspace() {
  const router = useRouter();
  const [messages, setMessages] = useState<ChatMessage[]>(readHistory);
  const [question, setQuestion] = useState("");
  const [inspectedMessageId, setInspectedMessageId] = useState<string | null>(
    null,
  );
  const [compactEvidenceOpen, setCompactEvidenceOpen] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const compactLayout = useMediaQuery("(max-width: 1100px)");
  const {
    ragMode,
    topK,
    evidenceOpen,
    activeSourceIndex,
    setRagMode,
    setTopK,
    setEvidenceOpen,
    selectSource,
  } = useWorkspaceStore();

  const health = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 15_000,
  });

  const chatMutation = useMutation({
    mutationFn: sendChat,
  });

  useEffect(() => {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(messages.slice(-30)));
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const inspectedMessage = useMemo(() => {
    const explicit = messages.find(
      (message) => message.id === inspectedMessageId,
    );
    return (
      explicit ??
      [...messages]
        .reverse()
        .find(
          (message) =>
            message.role === "assistant" && message.sources.length > 0,
        )
    );
  }, [inspectedMessageId, messages]);

  async function submitQuestion(value: string) {
    const cleanQuestion = value.trim();
    if (!cleanQuestion || chatMutation.isPending) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: cleanQuestion,
      sources: [],
    };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");

    try {
      const response = await chatMutation.mutateAsync({
        question: cleanQuestion,
        rag_mode: ragMode,
        top_k: topK,
      });
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        mode: response.mode_used,
        safetyWarning: response.safety_warning,
      };
      setMessages((current) => [...current, assistantMessage]);
      if (response.sources.length > 0) {
        setInspectedMessageId(assistantMessage.id);
        selectSource(0);
      }
    } catch {
      textareaRef.current?.focus();
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void submitQuestion(question);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitQuestion(question);
    }
  }

  function inspectCitation(messageId: string, index: number) {
    setInspectedMessageId(messageId);
    selectSource(index);
    if (compactLayout) setCompactEvidenceOpen(true);
  }

  function retryLastQuestion() {
    const lastQuestion = [...messages]
      .reverse()
      .find((message) => message.role === "user");
    if (lastQuestion) void submitQuestion(lastQuestion.content);
  }

  const hasMessages = messages.length > 0;
  const offline = health.isError;
  const showEvidence = compactLayout ? compactEvidenceOpen : evidenceOpen;

  function toggleEvidence() {
    if (compactLayout) {
      setCompactEvidenceOpen((open) => !open);
      return;
    }
    setEvidenceOpen(!evidenceOpen);
  }

  function closeEvidence() {
    if (compactLayout) {
      setCompactEvidenceOpen(false);
      return;
    }
    setEvidenceOpen(false);
  }

  function openEvidenceForMessage(messageId: string) {
    setInspectedMessageId(messageId);
    if (compactLayout) {
      setCompactEvidenceOpen(true);
      return;
    }
    setEvidenceOpen(true);
  }

  return (
    <main
      className={`chat-workspace ${
        showEvidence ? "chat-workspace--evidence" : ""
      }`}
    >
      <section className="chat-column">
        <header className="conversation-header">
          <div className="conversation-title">
            <span>MedTutor Library</span>
            <h1>Hỏi đáp theo bằng chứng</h1>
          </div>

          <div className="conversation-controls">
            <label className="rag-mode-control">
              <span className="sr-only">Chế độ RAG</span>
              <select
                value={ragMode}
                onChange={(event) =>
                  setRagMode(event.target.value as RagMode)
                }
                aria-label="Chế độ RAG"
              >
                {ragModes.map((mode) => (
                  <option key={mode.value} value={mode.value}>
                    {mode.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="top-k-control" title="Số chunk được truy xuất">
              <Settings2 size={15} aria-hidden="true" />
              <span className="sr-only">Top K</span>
              <select
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              >
                {[3, 5, 8, 10].map((value) => (
                  <option key={value} value={value}>
                    K={value}
                  </option>
                ))}
              </select>
            </label>

            <button
              className={`icon-button ${showEvidence ? "is-active" : ""}`}
              type="button"
              onClick={toggleEvidence}
              aria-label={
                showEvidence ? "Ẩn panel bằng chứng" : "Hiện panel bằng chứng"
              }
              aria-pressed={showEvidence}
              title="Bằng chứng"
            >
              <PanelRight size={18} />
            </button>
          </div>
        </header>

        {offline && (
          <div className="connection-alert" role="alert">
            <AlertTriangle size={17} aria-hidden="true" />
            <span>
              Backend đang offline. Khởi động FastAPI tại cổng 8000 để gửi câu
              hỏi.
            </span>
          </div>
        )}

        <div className="message-scroller">
          {!hasMessages ? (
            <div className="chat-empty">
              <div className="empty-symbol">
                <BookOpenCheck size={25} aria-hidden="true" />
              </div>
              <div>
                <div className="eyebrow">Grounded learning</div>
                <h2>Hỏi tài liệu, kiểm tra từng bằng chứng</h2>
                <p>
                  Câu trả lời chỉ dựa trên dữ liệu đã ingest. Mỗi citation liên
                  kết trực tiếp tới chunk được truy xuất.
                </p>
              </div>
              <div className="prompt-list" aria-label="Câu hỏi gợi ý">
                {examples.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => {
                      setQuestion(example);
                      textareaRef.current?.focus();
                    }}
                  >
                    <Sparkles size={15} aria-hidden="true" />
                    <span>{example}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) =>
                message.role === "user" ? (
                  <article className="user-message" key={message.id}>
                    <div className="message-author">Bạn</div>
                    <p>{message.content}</p>
                  </article>
                ) : (
                  <article
                    className={`assistant-message ${
                      message.safetyWarning
                        ? `assistant-message--${message.safetyWarning}`
                        : ""
                    }`}
                    key={message.id}
                  >
                    <header className="assistant-author">
                      <div className="assistant-mark" aria-hidden="true">
                        {message.safetyWarning === "emergency" ? (
                          <ShieldAlert size={16} />
                        ) : (
                          <BookOpenCheck size={16} />
                        )}
                      </div>
                      <div>
                        <strong>MedTutor</strong>
                        <span>
                          {message.safetyWarning === "emergency"
                            ? "Cảnh báo khẩn cấp"
                            : message.safetyWarning ===
                                "blocked_medical_advice"
                              ? "Giới hạn tư vấn y tế"
                              : `${message.mode ?? ragMode} RAG`}
                        </span>
                      </div>
                    </header>

                    {message.sources.length === 0 &&
                      !message.safetyWarning && (
                        <div className="no-context-label">
                          Không tìm thấy ngữ cảnh phù hợp trong kho tài liệu
                        </div>
                      )}

                    <AnswerContent
                      answer={message.content}
                      sourceCount={message.sources.length}
                      activeSourceIndex={
                        inspectedMessage?.id === message.id
                          ? activeSourceIndex
                          : -1
                      }
                      onCitationSelect={(index) =>
                        inspectCitation(message.id, index)
                      }
                    />

                    <footer className="answer-actions">
                      <button
                        type="button"
                        onClick={() =>
                          void navigator.clipboard.writeText(message.content)
                        }
                      >
                        <Copy size={14} aria-hidden="true" />
                        Sao chép
                      </button>
                      <button type="button" onClick={retryLastQuestion}>
                        <RotateCcw size={14} aria-hidden="true" />
                        Thử lại
                      </button>
                      {message.sources.length > 0 && (
                        <button
                          type="button"
                          onClick={() => openEvidenceForMessage(message.id)}
                        >
                          <PanelRight size={14} aria-hidden="true" />
                          {message.sources.length} nguồn
                        </button>
                      )}
                    </footer>
                  </article>
                ),
              )}

              {chatMutation.isPending && (
                <article className="assistant-message" aria-live="polite">
                  <header className="assistant-author">
                    <div className="assistant-mark">
                      <LoaderCircle
                        className="spin"
                        size={16}
                        aria-hidden="true"
                      />
                    </div>
                    <div>
                      <strong>MedTutor</strong>
                      <span>Đang truy xuất bằng chứng</span>
                    </div>
                  </header>
                  <div className="answer-skeleton">
                    <span />
                    <span />
                    <span />
                  </div>
                </article>
              )}

              {chatMutation.isError && (
                <div className="inline-error" role="alert">
                  <AlertTriangle size={17} aria-hidden="true" />
                  <span>{chatMutation.error.message}</span>
                </div>
              )}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <div className="composer-region">
          <form className="chat-composer" onSubmit={handleSubmit}>
            <div className="composer-actions">
              <button
                className="icon-button"
                type="button"
                onClick={() => router.push("/ingest")}
                aria-label="Đính kèm tài liệu"
                title="Mở trang ingest tài liệu"
              >
                <FilePlus2 size={18} />
              </button>
              <button
                className="icon-button"
                type="button"
                onClick={() => router.push("/ingest")}
                aria-label="Ghi âm câu hỏi"
                title="Mở trang ingest audio"
              >
                <Mic size={18} />
              </button>
            </div>
            <textarea
              ref={textareaRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              maxLength={3000}
              placeholder="Hỏi một câu dựa trên tài liệu đã ingest..."
              aria-label="Câu hỏi"
            />
            <button
              className="send-button"
              type="submit"
              disabled={!question.trim() || chatMutation.isPending || offline}
              aria-label="Gửi câu hỏi"
              title="Gửi"
            >
              {chatMutation.isPending ? (
                <LoaderCircle className="spin" size={18} />
              ) : (
                <Send size={18} />
              )}
            </button>
          </form>
          <p>
            Thông tin chỉ phục vụ học tập/tham khảo, không thay thế tư vấn của
            bác sĩ.
          </p>
        </div>
      </section>

      {showEvidence && (
        <>
          <button
            className={`evidence-scrim ${
              compactEvidenceOpen ? "evidence-scrim--compact-open" : ""
            }`}
            type="button"
            aria-label="Đóng bằng chứng"
            onClick={closeEvidence}
          />
          <EvidenceInspector
            sources={inspectedMessage?.sources ?? []}
            activeIndex={activeSourceIndex}
            onSelect={selectSource}
            onClose={closeEvidence}
            compactOpen={compactEvidenceOpen}
          />
        </>
      )}
    </main>
  );
}

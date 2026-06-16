"use client";

import ReactMarkdown from "react-markdown";

export function prepareAnswerMarkdown(answer: string): string {
  return answer
    .replace(/^(\d+\.\s[^:\n]+):$/gm, "### $1")
    .replace(/\[(\d+)\](?!\()/g, "[$1](citation:$1)");
}

export function AnswerContent({
  answer,
  sourceCount,
  activeSourceIndex,
  onCitationSelect,
}: {
  answer: string;
  sourceCount: number;
  activeSourceIndex: number;
  onCitationSelect: (index: number) => void;
}) {
  return (
    <div className="answer-content">
      <ReactMarkdown
        urlTransform={(url) => url}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith("citation:")) {
              const sourceNumber = Number(href.split(":")[1]);
              const sourceIndex = sourceNumber - 1;
              const missing =
                !Number.isInteger(sourceIndex) ||
                sourceIndex < 0 ||
                sourceIndex >= sourceCount;

              return (
                <button
                  className={`citation-button ${
                    sourceIndex === activeSourceIndex
                      ? "citation-button--active"
                      : ""
                  } ${missing ? "citation-button--missing" : ""}`}
                  type="button"
                  disabled={missing}
                  onClick={() => onCitationSelect(sourceIndex)}
                  aria-label={
                    missing
                      ? `Không tìm thấy nguồn ${sourceNumber}`
                      : `Xem nguồn ${sourceNumber}`
                  }
                  title={
                    missing
                      ? "Citation không có nguồn tương ứng"
                      : `Xem bằng chứng nguồn ${sourceNumber}`
                  }
                >
                  [{sourceNumber}]
                </button>
              );
            }

            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {prepareAnswerMarkdown(answer)}
      </ReactMarkdown>
    </div>
  );
}

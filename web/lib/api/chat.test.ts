import { describe, expect, it } from "vitest";

import { enrichSources, type ChatResponse } from "@/lib/api/chat";

describe("enrichSources", () => {
  it("joins retrieved chunk text to citation metadata", () => {
    const response: ChatResponse = {
      answer: "Answer [1]",
      mode_used: "hybrid",
      safety_warning: null,
      sources: [
        {
          chunk_id: "chunk-1",
          title: "Stroke guide",
          source_path: "stroke.pdf",
          source_type: "pdf",
          score: 0.91,
          page_number: 4,
          section_title: "FAST",
        },
      ],
      retrieved_chunks: [
        {
          chunk: {
            chunk_id: "chunk-1",
            text: "FAST includes face, arm and speech signs.",
          },
          score: 0.91,
        },
      ],
    };

    expect(enrichSources(response).sources[0].excerpt).toBe(
      "FAST includes face, arm and speech signs.",
    );
  });
});

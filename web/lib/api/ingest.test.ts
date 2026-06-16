import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ingestUrl,
  uploadDocument,
  uploadMedia,
} from "@/lib/api/ingest";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ingest API", () => {
  it("uploads documents as multipart without forcing a JSON content type", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          doc_id: "doc-1",
          title: "stroke",
          chunks_added: 2,
          source_path: "data/raw/stroke.md",
          source_type: "md",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    const file = new File(["# FAST"], "stroke.md", {
      type: "text/markdown",
    });

    await uploadDocument(file);

    const [, init] = fetchMock.mock.calls[0];
    expect(init?.body).toBeInstanceOf(FormData);
    expect(new Headers(init?.headers).has("Content-Type")).toBe(false);
  });

  it("routes image, audio and video files to their dedicated endpoints", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          doc_id: "media-1",
          title: "lecture",
          chunks_added: 1,
          source_path: "data/raw/lecture.wav",
          source_type: "audio",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    await uploadMedia(
      new File(["audio"], "lecture.wav", { type: "audio/wav" }),
      "audio",
    );

    expect(fetchMock.mock.calls[0][0]).toContain("/voice/ingest");
  });

  it("sends public URLs as JSON", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          doc_id: "url-1",
          title: "guideline",
          chunks_added: 1,
          source_path: "data/raw/url.html",
          source_type: "html",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    await ingestUrl("https://example.org/guideline");

    const [, init] = fetchMock.mock.calls[0];
    expect(new Headers(init?.headers).get("Content-Type")).toBe(
      "application/json",
    );
    expect(init?.body).toBe(
      JSON.stringify({ url: "https://example.org/guideline" }),
    );
  });
});

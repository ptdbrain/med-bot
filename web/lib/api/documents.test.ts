import { afterEach, describe, expect, it, vi } from "vitest";

import { deleteDocument, getDocuments } from "@/lib/api/documents";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("documents API", () => {
  it("loads the document library", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ documents: [], total: 0 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await getDocuments();

    expect(fetchMock.mock.calls[0][0]).toContain("/documents");
    expect(fetchMock.mock.calls[0][1]?.method).toBeUndefined();
  });

  it("deletes an encoded document id", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ doc_id: "doc/1", deleted: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await deleteDocument("doc/1");

    expect(fetchMock.mock.calls[0][0]).toContain("/documents/doc%2F1");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("DELETE");
  });
});

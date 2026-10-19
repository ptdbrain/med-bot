import { afterEach, describe, expect, it, vi } from "vitest";
import { createIngestJob, getIngestJob, retryIngestJob } from "./ingest";

afterEach(() => vi.restoreAllMocks());
describe("ingest jobs", () => {
  it("submits a document and retrieves its durable job", async () => {
    const job = { job_id: "j1", status: "queued", filename: "notes.pdf", report: {}, error: null };
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify(job), { status: 202 }));
    expect((await createIngestJob(new File(["pdf"], "notes.pdf"))).job_id).toBe("j1");
    expect(fetch.mock.calls[0][0]).toContain("/ingest/jobs");
    expect((await getIngestJob("j1")).status).toBe("queued");
    await retryIngestJob("j1");
    expect(fetch.mock.calls[2][0]).toContain("/j1/retry");
  });
});

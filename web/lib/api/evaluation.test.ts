import { afterEach, describe, expect, it, vi } from "vitest";

import {
  describeEvaluationDetail,
  runEvaluation,
} from "@/lib/api/evaluation";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("evaluation API", () => {
  it("runs the default evaluation dataset", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          metrics: [{ name: "hit_at_k", value: 1 }],
          details: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    const result = await runEvaluation(5);

    expect(result.metrics[0].name).toBe("hit_at_k");
    expect(fetchMock.mock.calls[0][1]?.body).toBe(
      JSON.stringify({
        dataset_path: "app/evaluation/eval_dataset.json",
        top_k: 5,
      }),
    );
  });

  it("describes safety and retrieval outcomes for the audit table", () => {
    expect(
      describeEvaluationDetail({
        category: "safety",
        expected: "emergency",
        actual_warning: "emergency",
      }),
    ).toBe("emergency → emergency");
    expect(
      describeEvaluationDetail({
        category: "retrieval",
        expected_source: "stroke",
        rank: 1,
      }),
    ).toBe("stroke, hạng 1");
  });
});

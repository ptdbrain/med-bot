from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.benchmark import run_benchmark
from app.evaluation.claim_eval import evaluate_claims


def test_claim_eval_all_required_claims_satisfied():
    answer = (
        "1. Câu trả lời ngắn gọn: Tiêu sợi huyết rtPA đường tĩnh mạch được chỉ định "
        "trong vòng 4.5 giờ từ khi khởi phát triệu chứng.\n"
        "2. Giải thích chi tiết: Tuyệt đối không dùng kháng đông hoặc aspirin trong vòng 24 giờ đầu sau rtPA.\n"
        "3. Nguồn đã dùng: [1]\n4. Cảnh báo y tế: Tham vấn bác sĩ."
    )
    required = [
        "Tiêu sợi huyết rtPA đường tĩnh mạch được chỉ định trong vòng 4.5 giờ từ khi khởi phát triệu chứng.",
        "Tuyệt đối không dùng kháng đông hoặc aspirin trong vòng 24 giờ đầu sau khi truyền rtPA.",
    ]
    unsupported = [
        "rtPA dùng tự do đến 24 giờ",
        "Aspirin nên uống ngay lập tức",
    ]

    res = evaluate_claims(answer, required, unsupported, expected="answerable")
    assert res["claim_recall"] == 1.0
    assert res["unsupported_claims_detected"] == 0
    assert res["faithfulness_score"] == 1.0
    assert res["passed"] is True


def test_claim_eval_penalizes_hallucinated_unsupported_claim():
    answer = (
        "Tiêu sợi huyết rtPA dùng trong 4.5 giờ. "
        "Tuy nhiên cho bệnh nhân uống aspirin ngay lập tức không cần đợi 24 giờ."
    )
    required = [
        "Tiêu sợi huyết rtPA đường tĩnh mạch được chỉ định trong vòng 4.5 giờ",
    ]
    unsupported = [
        "cho bệnh nhân uống aspirin ngay lập tức",
    ]

    res = evaluate_claims(answer, required, unsupported, expected="answerable")
    assert res["claim_recall"] == 1.0
    assert res["unsupported_claims_detected"] == 1
    assert res["faithfulness_score"] < 1.0
    assert res["passed"] is False


def test_claim_eval_handles_insufficient_context():
    refusal_answer = "Tài liệu hiện tại không đủ thông tin để trả lời câu hỏi này."
    hallucinating_answer = "Thuốc mới phát minh XYZ-999 dùng liều 500mg mỗi ngày chữa khỏi bệnh."

    good_res = evaluate_claims(refusal_answer, [], ["XYZ-999 dùng liều 500mg"], expected="insufficient_context")
    assert good_res["faithfulness_score"] == 1.0
    assert good_res["passed"] is True

    bad_res = evaluate_claims(hallucinating_answer, [], ["XYZ-999 dùng liều 500mg"], expected="insufficient_context")
    assert bad_res["faithfulness_score"] == 0.0
    assert bad_res["passed"] is False


def test_benchmark_faithfulness_evaluation_with_generation(tmp_path: Path):
    config_path = tmp_path / "config_gen.json"
    output_path = tmp_path / "output_gen.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/reviewed_education/corpus/manifest.json",
                "cases": "evaluation/reviewed_education/cases",
                "modes": ["bm25"],
                "generate": True,
            }
        ),
        encoding="utf-8",
    )

    report = run_benchmark(config_path, "dev", output_path, limit=2)
    assert report["faithfulness"]["status"] == "evaluated"
    assert "mean_faithfulness_score" in report["faithfulness"]
    assert "mean_claim_recall" in report["faithfulness"]
    assert "mean_claim_recall" in report["summary"]["bm25"]
    assert "mean_faithfulness_score" in report["summary"]["bm25"]

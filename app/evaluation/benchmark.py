from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

from app.config import resolve_profile
from app.evaluation.dataset import load_cases, validate_cases
from app.evaluation.metrics import percentile, ranking_metrics
from app.rag.service import RAGService
from app.retrieval.index_manifest import embedding_fingerprint
from app.retrieval.vector_store import normalize_text


def _normalized(text):
    return " ".join(normalize_text(text).split())


def resolve_evidence(results, evidence):
    identifiers = []
    for result in results:
        matched = [
            key for key, value in evidence.items() if _normalized(value["quote"]) in _normalized(result.chunk.text)
        ]
        identifiers.append(matched[0] if matched else "unlabelled:" + result.chunk.chunk_id)
    return identifiers


def run_benchmark(config: Path, split: str, output: Path, *, allow_draft=False, limit=None):
    raw = config.read_bytes()
    options = json.loads(raw)
    manifest_path = Path(options.get("corpus", "evaluation/corpus/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_cases = [
        c
        for name in ("dev", "test")
        for c in load_cases(Path(options.get("cases", "evaluation/cases")) / f"{name}.jsonl")
    ]
    problems = validate_cases(all_cases, manifest, require_review=not allow_draft)
    if problems:
        raise ValueError("Invalid evaluation dataset: " + ", ".join(problems[:10]))
    settings = resolve_profile(options.get("profile", "demo"), options.get("overrides", {}))
    # Benchmark data cannot overwrite the ordinary application index.
    data_root = settings.data_dir.resolve()
    if data_root == Path("data").resolve():
        raise ValueError("Benchmark requires a separate DATA_DIR")
    service = RAGService(settings)
    for source in sorted({value["source_path"] for value in manifest["evidence"].values()}):
        path = Path(source)
        service.ingest_file(path.name, path.read_bytes())
    cases = [case for case in all_cases if case.split == split]
    if limit is not None:
        cases = cases[:limit]
    modes = options.get("modes", ["bm25", "dense", "hybrid", "rerank"])
    details = []
    for case in cases:
        for mode in modes:
            started = time.perf_counter()
            row = {
                "case_id": case.case_id,
                "category": case.category,
                "language": case.query_language,
                "mode": mode,
                "expected": case.expected,
                "status": "ok",
            }
            try:
                if case.expected in {"emergency", "blocked_medical_advice"}:
                    answer = service.chat(case.question, rag_mode="agentic")
                    row["safety_pass"] = answer.safety_warning == case.expected
                else:
                    if mode == "bm25":
                        results = service.bm25_store.search(case.question, top_k=10)
                    elif mode == "dense":
                        results = service.vector_store.search(case.question, top_k=10)
                    else:
                        results = service.retrieve(case.question, top_k=10, rag_mode=mode)
                    predicted = resolve_evidence(results, manifest["evidence"])
                    row["retrieved_evidence"] = predicted
                    row["retrieved_chunks"] = [r.chunk.chunk_id for r in results]
                    if case.expected == "answerable":
                        relevance = dict.fromkeys(case.relevant_evidence_ids, 1)
                        row["at_5"] = ranking_metrics(predicted, relevance, 5)
                        row["at_10"] = ranking_metrics(predicted, relevance, 10)
                    decisions = [service.relevance_policy.evaluate(case.question, r) for r in results]
                    row["refused"] = not any(d.accepted for d in decisions)
                    row["reasons"] = [d.reasons for d in decisions]
                    if options.get("generate", False):
                        answer = service.chat(case.question, rag_mode="rerank")
                        row["answer"] = answer.answer
                        row["answer_validation"] = answer.validation
            except Exception as exc:  # noqa: BLE001 - Record a failed case rather than dropping its denominator.
                row["status"], row["error"] = "failed", type(exc).__name__
            row["seconds"] = time.perf_counter() - started
            details.append(row)
    summary = {}
    for mode in modes:
        rows = [row for row in details if row["mode"] == mode]
        answerable = [row for row in rows if row["expected"] == "answerable"]
        metrics = {}
        for k in (5, 10):
            for metric in ("recall", "hit", "mrr", "ndcg"):
                metrics[f"{metric}_at_{k}"] = sum(row.get(f"at_{k}", {}).get(metric, 0) for row in answerable) / max(
                    1, len(answerable)
                )
        unknown = [row for row in rows if row["expected"] == "insufficient_context"]
        refusals = [
            row for row in rows if row.get("refused") and row["expected"] in {"answerable", "insufficient_context"}
        ]
        correct_refusals = sum(row.get("refused", False) for row in unknown)
        metrics["unanswerable_precision"] = correct_refusals / len(refusals) if refusals else None
        metrics["unanswerable_recall"] = correct_refusals / len(unknown) if unknown else None
        metrics["answerable_coverage"] = sum(not row.get("refused", True) for row in answerable) / max(
            1, len(answerable)
        )
        metrics.update(
            cases=len(rows),
            answerable=len(answerable),
            failures=sum(row["status"] == "failed" for row in rows),
            p50_seconds=percentile([r["seconds"] for r in rows], 0.5),
            p95_seconds=percentile([r["seconds"] for r in rows], 0.95),
        )
        summary[mode] = metrics
    report = {
        "split": split,
        "case_count": len(cases),
        "dataset_reviewed": all(c.reviewed_by for c in cases),
        "quality_gate": (
            "not_evaluated_unreviewed" if not all(c.reviewed_by for c in cases) else "requires_answer_and_human_audit"
        ),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "embedding_fingerprint": embedding_fingerprint(settings),
        "llm_provider": settings.llm_provider,
        "reranker_provider": settings.reranker_provider,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "faithfulness": {"status": "not_evaluated", "reason": "Human claim/evidence audit required"},
        "summary": summary,
        "details": details,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

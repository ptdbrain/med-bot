from __future__ import annotations

import argparse

from app.config import get_settings
from app.evaluation.answer_eval import evaluate_answers
from app.evaluation.retrieval_eval import evaluate_retrieval
from app.evaluation.safety_eval import evaluate_safety
from app.rag.service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the small draft smoke evaluation set")
    parser.add_argument("--split", choices=("dev",), default="dev")
    args = parser.parse_args()
    settings = get_settings()
    service = RAGService()
    dataset = "app/evaluation/eval_dataset.json"
    metrics = []
    metrics.extend(evaluate_retrieval(service, dataset, top_k=5))
    metrics.extend(evaluate_answers(service, dataset))
    metrics.extend(evaluate_safety(service, dataset))
    print(f"DRAFT / {args.split} / smoke cases; not a reviewed or held-out benchmark")
    print(
        "providers: "
        f"embedding={settings.embedding_provider}, vector_store={settings.vector_store_provider}, "
        f"reranker={settings.reranker_provider}, llm={settings.llm_provider}"
    )
    for metric in metrics:
        print(f"{metric.name}: {metric.value:.3f}")


if __name__ == "__main__":
    main()

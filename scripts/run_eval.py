from __future__ import annotations

from app.evaluation.answer_eval import evaluate_answers
from app.evaluation.retrieval_eval import evaluate_retrieval
from app.evaluation.safety_eval import evaluate_safety
from app.rag.service import RAGService


def main() -> None:
    service = RAGService()
    dataset = "app/evaluation/eval_dataset.json"
    metrics = []
    metrics.extend(evaluate_retrieval(service, dataset, top_k=5))
    metrics.extend(evaluate_answers(service, dataset))
    metrics.extend(evaluate_safety(service, dataset))
    for metric in metrics:
        print(f"{metric.name}: {metric.value:.3f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from fastapi import APIRouter, Request

from app.evaluation.answer_eval import evaluate_answers_with_details
from app.evaluation.retrieval_eval import evaluate_retrieval_with_details
from app.evaluation.safety_eval import evaluate_safety_with_details
from app.schemas.evaluation import EvalRequest, EvalResponse

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.post("/run", response_model=EvalResponse)
def run_eval(payload: EvalRequest, request: Request) -> EvalResponse:
    service = request.app.state.rag_service
    metrics, details = [], []
    for evaluator, args in (
        (
            evaluate_retrieval_with_details,
            (service, payload.dataset_path, payload.top_k),
        ),
        (evaluate_answers_with_details, (service, payload.dataset_path)),
        (evaluate_safety_with_details, (service, payload.dataset_path)),
    ):
        evaluator_metrics, evaluator_details = evaluator(*args)
        metrics.extend(evaluator_metrics)
        details.extend(evaluator_details)
    return EvalResponse(metrics=metrics, details=details)

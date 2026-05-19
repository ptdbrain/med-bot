from __future__ import annotations

from fastapi import APIRouter, Request

from app.evaluation.answer_eval import evaluate_answers
from app.evaluation.retrieval_eval import evaluate_retrieval
from app.evaluation.safety_eval import evaluate_safety
from app.schemas.evaluation import EvalRequest, EvalResponse

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.post("/run", response_model=EvalResponse)
def run_eval(payload: EvalRequest, request: Request) -> EvalResponse:
    service = request.app.state.rag_service
    metrics, details = [], []
    metrics.extend(evaluate_retrieval(service, payload.dataset_path, payload.top_k))
    metrics.extend(evaluate_answers(service, payload.dataset_path))
    metrics.extend(evaluate_safety(service, payload.dataset_path))
    return EvalResponse(metrics=metrics, details=details)

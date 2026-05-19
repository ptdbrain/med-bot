from __future__ import annotations

from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    dataset_path: str = "app/evaluation/eval_dataset.json"
    top_k: int = Field(default=5, ge=1, le=20)


class EvalMetric(BaseModel):
    name: str
    value: float


class EvalResponse(BaseModel):
    metrics: list[EvalMetric]
    details: list[dict] = Field(default_factory=list)

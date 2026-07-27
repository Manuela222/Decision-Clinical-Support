"""Multi-label evaluation results (Phase 14): per-admission detail plus
micro/macro aggregates, shared by baseline, trained model, and agent so all
three are scored with the identical metric function."""
from typing import Optional

from pydantic import BaseModel, Field

from .recommendation import RecommendationMethod


class PerAdmissionEvaluation(BaseModel):
    subject_id: int
    hadm_id: int
    method: RecommendationMethod
    predicted_classes: list[str] = Field(default_factory=list)
    ground_truth_classes: list[str] = Field(default_factory=list)
    matched_classes: list[str] = Field(default_factory=list)
    missed_classes: list[str] = Field(default_factory=list, description="ground truth, not predicted")
    extra_classes: list[str] = Field(default_factory=list, description="predicted, not in ground truth")
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    method: RecommendationMethod
    n_admissions_evaluated: int

    micro_precision: float = Field(..., ge=0.0, le=1.0)
    micro_recall: float = Field(..., ge=0.0, le=1.0)
    micro_f1: float = Field(..., ge=0.0, le=1.0)

    macro_precision: float = Field(..., ge=0.0, le=1.0)
    macro_recall: float = Field(..., ge=0.0, le=1.0)
    macro_f1: float = Field(..., ge=0.0, le=1.0)

    per_admission: list[PerAdmissionEvaluation] = Field(default_factory=list)
    notes: Optional[str] = None

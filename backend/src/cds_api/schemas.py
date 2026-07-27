"""API-layer request/response schemas. Distinct from `cds.schemas` (the
domain schemas) — these exist only to shape what crosses the HTTP boundary."""
from typing import List, Optional

from pydantic import BaseModel

from cds.schemas import EvaluationResult


class CohortPatientResponse(BaseModel):
    subject_id: int
    hadm_id: int
    age: int
    gender: str
    admission_reason: str
    hypertension_status: str


class PredictRequest(BaseModel):
    subject_id: int
    hadm_id: int


class NewPatientRequest(BaseModel):
    """Phase 17's "new patient" form: current_medication_classes must be
    drawn from the fixed medication-class vocabulary (select-from-list on
    the frontend) — the service layer rejects anything else, it does not
    silently coerce free text."""

    age: int
    gender: str
    admission_reason: str
    hypertension_status: str
    current_medication_classes: List[str] = []
    recent_creatinine: Optional[float] = None


class EvaluateRequest(BaseModel):
    hadm_ids: Optional[List[int]] = None
    include_agent: bool = False


class EvaluateResponse(BaseModel):
    comparison_table: List[dict]
    evaluations: List[EvaluationResult]

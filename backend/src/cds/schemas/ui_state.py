"""UIState: the JSON-serializable contract between backend responses and the
two frontend pages (Phase 17) — existing test-set patient comparison, and the
new/synthetic-patient agent-only page."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .clinical_state import ClinicalState
from .recommendation import RecommendationResult


class UIPage(str, Enum):
    EXISTING_PATIENT = "existing_patient"
    NEW_PATIENT = "new_patient"


class RequestStatus(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    SUCCESS = "success"
    ERROR = "error"


class MethodResultState(BaseModel):
    status: RequestStatus = RequestStatus.IDLE
    result: Optional[RecommendationResult] = None
    error_message: Optional[str] = None


class UIState(BaseModel):
    page: UIPage
    selected_subject_id: Optional[int] = None
    selected_hadm_id: Optional[int] = None

    clinical_state: Optional[ClinicalState] = None

    baseline_result: MethodResultState = Field(default_factory=MethodResultState)
    trained_model_result: MethodResultState = Field(default_factory=MethodResultState)
    agent_result: MethodResultState = Field(default_factory=MethodResultState)

    ground_truth_medications: list[str] = Field(
        default_factory=list, description="Actual discharge medication classes, for the existing-patient page only"
    )

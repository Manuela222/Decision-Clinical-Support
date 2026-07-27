"""Multi-label medication recommendations, produced by any of the three methods
(baseline, trained model, agent) and rendered through the same UI/report shape."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .reasoning import ReasoningStep
from .safety import SafetyWarning


class RecommendationMethod(str, Enum):
    BASELINE = "baseline"
    TRAINED_MODEL = "trained_model"
    AGENT = "agent"


class MedicationAction(str, Enum):
    CONTINUE = "continue"
    START = "start"
    STOP = "stop"
    ADJUST = "adjust"
    AVOID = "avoid"


class MedicationRecommendation(BaseModel):
    """One recommended (or rejected) medication class, with its own rationale."""

    medication_class: str
    action: MedicationAction
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

    hypertension_compatible: bool = Field(
        ..., description="Explicit compatibility verdict against the patient's hypertension status"
    )
    hypertension_reasoning: str = Field(
        ..., description="Required, concise explanation of the hypertension-compatibility verdict"
    )


class RecommendationResult(BaseModel):
    subject_id: int
    hadm_id: int
    method: RecommendationMethod
    recommended_medications: list[MedicationRecommendation] = Field(default_factory=list)
    safety_warnings: list[SafetyWarning] = Field(default_factory=list)
    reasoning_trace: list[ReasoningStep] = Field(
        default_factory=list, description="Populated for method=agent; empty for baseline/trained_model"
    )
    model_version: Optional[str] = Field(default=None, description="e.g. trained-model artifact id or agent model name")
    generated_at: datetime

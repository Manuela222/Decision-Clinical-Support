"""Phase 15: the explainability/XAI report schema. Purely descriptive of an
existing RecommendationResult — this module never generates a new
recommendation, only explains one already produced by Phase 8/9/13."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from ..retrieval import SimilarPatientProfile
from ..schemas import ClinicalState, EvidenceItem, RecommendationMethod, SafetyWarning


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class MedicationCandidate(BaseModel):
    medication_class: str
    accepted: bool = Field(..., description="True if this class made it into the final recommendation")
    score: Optional[float] = Field(default=None, description="frequency (baseline) or predicted probability (model), if available")
    reason: Optional[str] = Field(default=None, description="Why rejected, when known; None for accepted candidates")


class ExplainabilityReport(BaseModel):
    subject_id: int
    hadm_id: int
    method: RecommendationMethod

    patient_profile: ClinicalState
    retrieved_similar_profiles: List[SimilarPatientProfile] = Field(default_factory=list)
    feature_importances: List[FeatureImportance] = Field(default_factory=list)
    medication_candidates: List[MedicationCandidate] = Field(default_factory=list)
    safety_warnings: List[SafetyWarning] = Field(default_factory=list)
    evidence_citations: List[EvidenceItem] = Field(default_factory=list)

    ground_truth_medications: List[str] = Field(default_factory=list)
    matched_classes: List[str] = Field(default_factory=list)
    missed_classes: List[str] = Field(default_factory=list)
    extra_classes: List[str] = Field(default_factory=list)

    generated_at: datetime

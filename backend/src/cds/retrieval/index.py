"""Phase 11: embedding index over TRAIN-split patient ClinicalStates only.

*** Test-split patients must NEVER be passed to build_patient_profile_index. ***
This module has no way to verify that on its own (it just indexes whatever
DataFrame/list it's given) — same as Phase 9's `train_model`, the guarantee
depends on the caller only ever passing the Phase 6 TRAIN split here. Doing
otherwise would let the agent's RAG signal (Phase 13) retrieve a test
patient's own ground-truth discharge medications, invalidating Phase 14's
evaluation.

The retrieved "actual discharge medications" for each similar profile *is*
the core RAG signal Phase 13's agent reasons from — that's expected and
fine, because it's always a *different*, train-split patient's already-known
outcome, never the query patient's own answer.
"""
from dataclasses import dataclass
from typing import Any, List

from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from ..model.features import ALL_FEATURE_COLUMNS, build_structured_features
from ..schemas import ClinicalState, HypertensionStatus
from .encoder import build_profile_encoder


class SimilarPatientProfile(BaseModel):
    subject_id: int
    hadm_id: int
    similarity: float
    admission_reason: str
    hypertension_status: HypertensionStatus
    ground_truth_medication_classes: List[str]
    evidence_id: str


@dataclass
class PatientProfileIndex:
    subject_ids: List[int]
    hadm_ids: List[int]
    clinical_states: List[ClinicalState]
    ground_truth_classes: List[List[str]]
    encoder: Any  # fitted sklearn ColumnTransformer
    vectors: Any  # fitted encoder's output over the indexed patients


def build_patient_profile_index(
    train_clinical_states: List[ClinicalState],
    train_ground_truth_classes: List[List[str]],
) -> PatientProfileIndex:
    """Build the RAG index. Pass ONLY Phase 6 TRAIN-split patients — see
    module docstring."""
    if len(train_clinical_states) != len(train_ground_truth_classes):
        raise ValueError(
            f"train_clinical_states ({len(train_clinical_states)}) and train_ground_truth_classes "
            f"({len(train_ground_truth_classes)}) must be the same length."
        )
    if not train_clinical_states:
        raise ValueError("Cannot build a patient profile index over zero patients.")

    features_df = build_structured_features(train_clinical_states)
    encoder = build_profile_encoder()
    vectors = encoder.fit_transform(features_df[ALL_FEATURE_COLUMNS])

    return PatientProfileIndex(
        subject_ids=[cs.subject_id for cs in train_clinical_states],
        hadm_ids=[cs.hadm_id for cs in train_clinical_states],
        clinical_states=list(train_clinical_states),
        ground_truth_classes=[list(c) for c in train_ground_truth_classes],
        encoder=encoder,
        vectors=vectors,
    )


def find_similar_patient_profiles(
    query_profile: ClinicalState, index: PatientProfileIndex, top_k: int = 5
) -> List[SimilarPatientProfile]:
    """Cosine-similarity search over the train-split index. Returns the
    `top_k` most similar train-split patients along with their actual
    discharge medication classes — the core RAG signal for Phase 13."""
    query_df = build_structured_features([query_profile])[ALL_FEATURE_COLUMNS]
    query_vector = index.encoder.transform(query_df)
    similarities = cosine_similarity(query_vector, index.vectors)[0]

    ranked = sorted(range(len(similarities)), key=lambda i: -similarities[i])[:top_k]
    return [
        SimilarPatientProfile(
            subject_id=index.subject_ids[i],
            hadm_id=index.hadm_ids[i],
            similarity=float(similarities[i]),
            admission_reason=index.clinical_states[i].admission_reason,
            hypertension_status=index.clinical_states[i].hypertension_status,
            ground_truth_medication_classes=list(index.ground_truth_classes[i]),
            evidence_id=f"train-profile-{index.subject_ids[i]}-{index.hadm_ids[i]}",
        )
        for i in ranked
    ]

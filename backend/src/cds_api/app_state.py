"""Phase 16: the API's dependency-injected state container.

Holds every precomputed artifact the routes need (test cohort, timelines,
clinical states, ground truth, baseline stats, trained model, RAG index,
LLM provider factory) so route handlers never touch raw MIMIC data or
retrain/rebuild anything per-request — they just look things up here and
call a `cds` business-logic function.

Building the REAL artifacts (loading the actual MIMIC_III_10k extract,
training the real model, indexing the real train split) is a deployment/
wiring concern outside this phase's scope, same as Phase 8/9's own tests —
this module only defines the container shape plus a `build_fake_app_state()`
used by the test suite (fake/synthetic data, per the spec for this phase).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from cds.agent import LLMProvider, MockLLMProvider, OpenAIProvider
from cds.baseline import DiagnosisMedicationStats, compute_diagnosis_medication_stats
from cds.model import TrainedModelArtifact, train_model
from cds.retrieval import PatientProfileIndex, build_patient_profile_index
from cds.schemas import ClinicalState, PatientTimeline

PatientKey = Tuple[int, int]  # (subject_id, hadm_id)


@dataclass
class CohortEntry:
    subject_id: int
    hadm_id: int
    age: int
    gender: str
    admission_reason: str
    hypertension_status: str
    split: str  # "train" or "test"


@dataclass
class AppState:
    cohort: List[CohortEntry]
    timelines: Dict[PatientKey, PatientTimeline]
    clinical_states: Dict[PatientKey, ClinicalState]
    ground_truth_classes: Dict[PatientKey, List[str]]
    diagnosis_medication_stats: DiagnosisMedicationStats
    trained_model_artifact: TrainedModelArtifact
    patient_profile_index: PatientProfileIndex
    llm_provider_factory: Callable[[], LLMProvider] = field(default=lambda: OpenAIProvider())

    def test_cohort(self) -> List[CohortEntry]:
        return [c for c in self.cohort if c.split == "test"]

    def get_clinical_state(self, subject_id: int, hadm_id: int) -> Optional[ClinicalState]:
        return self.clinical_states.get((subject_id, hadm_id))

    def get_timeline(self, subject_id: int, hadm_id: int) -> Optional[PatientTimeline]:
        return self.timelines.get((subject_id, hadm_id))

    def get_ground_truth(self, subject_id: int, hadm_id: int) -> List[str]:
        return self.ground_truth_classes.get((subject_id, hadm_id), [])


def build_fake_app_state(llm_provider_factory: Optional[Callable[[], LLMProvider]] = None) -> AppState:
    """Build a small, fully-synthetic AppState for tests/demos — no real
    MIMIC data touched anywhere in this function."""
    from datetime import datetime, timezone

    from cds.schemas import Gender, HypertensionStatus, LabValueSummary

    def _clinical_state(idx: int, reason: str, split_seed: int) -> ClinicalState:
        return ClinicalState(
            subject_id=idx,
            hadm_id=1000 + idx,
            age=40 + (idx % 40),
            gender=Gender.F if idx % 2 == 0 else Gender.M,
            admission_reason=reason,
            hypertension_status=HypertensionStatus.CONFIRMED_CHRONIC,
            hypertension_evidence=[],
            recent_blood_pressure_labs=[],
            current_antihypertensive_medications=[],
            active_conditions=[reason, "Hypertension"],
            chronic_conditions=[reason, "Hypertension"],
            recent_labs=[
                LabValueSummary(
                    itemid=1, label="Creatinine", value=0.6 + 0.01 * split_seed, unit="mg/dL",
                    flag=None, charttime=None, evidence_id=f"ev-{idx}",
                )
            ],
            current_medications=[],
            prior_medications=[],
            relevant_note_sections=[],
            missing_information=[],
            generated_at=datetime.now(timezone.utc),
        )

    reasons_and_labels = [
        ("Cardiac arrhythmias", ["antiarrhythmic"]),
        ("Diabetes, uncomplicated", ["antidiabetic - biguanide"]),
    ]

    train_states, train_labels = [], []
    for i in range(1, 21):
        reason, labels = reasons_and_labels[i % 2]
        train_states.append(_clinical_state(i, reason, i))
        train_labels.append(list(labels))

    test_states, test_labels = [], []
    for i in range(21, 26):
        reason, labels = reasons_and_labels[i % 2]
        test_states.append(_clinical_state(i, reason, i))
        test_labels.append(list(labels))

    all_states = train_states + test_states
    all_labels = train_labels + test_labels

    cohort = [
        CohortEntry(
            subject_id=cs.subject_id, hadm_id=cs.hadm_id, age=cs.age, gender=cs.gender.value,
            admission_reason=cs.admission_reason, hypertension_status=cs.hypertension_status.value,
            split="train" if cs in train_states else "test",
        )
        for cs in all_states
    ]
    clinical_states = {(cs.subject_id, cs.hadm_id): cs for cs in all_states}
    timelines = {
        (cs.subject_id, cs.hadm_id): PatientTimeline(
            subject_id=cs.subject_id, hadm_id=cs.hadm_id, events=[], generated_at=datetime.now(timezone.utc)
        )
        for cs in all_states
    }
    ground_truth = {(cs.subject_id, cs.hadm_id): labels for cs, labels in zip(all_states, all_labels)}

    diagnosis_medication_stats = compute_diagnosis_medication_stats(
        [cs.admission_reason for cs in train_states], train_labels
    )
    trained_model_artifact = train_model(train_states, train_labels, seed=42)
    patient_profile_index = build_patient_profile_index(train_states, train_labels)

    return AppState(
        cohort=cohort,
        timelines=timelines,
        clinical_states=clinical_states,
        ground_truth_classes=ground_truth,
        diagnosis_medication_stats=diagnosis_medication_stats,
        trained_model_artifact=trained_model_artifact,
        patient_profile_index=patient_profile_index,
        llm_provider_factory=llm_provider_factory or (lambda: MockLLMProvider([])),
    )

"""ClinicalState: a compact, deterministic-heuristic summary of a patient/admission.

Built without any LLM involvement (Phase 5). Every downstream module (baseline,
trained model, agent, safety checker) reads from this one object, so it must
explicitly surface hypertension status and hypertension-relevant values rather
than burying them in free text.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .common import EvidenceItem


class HypertensionStatus(str, Enum):
    CONFIRMED_CHRONIC = "confirmed_chronic"
    SUSPECTED = "suspected"
    NOT_PRESENT = "not_present"
    UNKNOWN = "unknown"


class Gender(str, Enum):
    M = "M"
    F = "F"


class LabValueSummary(BaseModel):
    """One recent, relevant lab result, reduced to what a clinician needs to see."""

    itemid: int
    label: str
    value: Optional[float] = None
    unit: Optional[str] = None
    flag: Optional[str] = Field(default=None, description="e.g. 'abnormal', or MIMIC's raw FLAG value")
    charttime: Optional[datetime] = None
    evidence_id: str


class MedicationMention(BaseModel):
    """A medication associated with the patient (current or prior), post-normalization."""

    normalized_name: str
    medication_class: str
    is_antihypertensive: bool = False
    evidence_id: str


class NoteSection(BaseModel):
    """A relevant excerpt from a clinical note, kept short and cited."""

    category: str
    section_title: Optional[str] = None
    excerpt: str
    evidence_id: str


class MissingInformationFlag(str, Enum):
    DISCHARGE_SUMMARY_MISSING = "discharge_summary_missing"
    LABS_MISSING = "labs_missing"
    NOTES_MISSING = "notes_missing"
    PRIOR_ADMISSIONS_MISSING = "prior_admissions_missing"
    HYPERTENSION_LABS_MISSING = "hypertension_labs_missing"


class ClinicalState(BaseModel):
    subject_id: int
    hadm_id: int
    age: int
    gender: Gender

    admission_reason: str = Field(..., description="Primary/admission condition group, e.g. 'Cardiac arrhythmias'")

    # --- hypertension-specific fields (explicit per project requirement) ---
    hypertension_status: HypertensionStatus
    hypertension_evidence: list[EvidenceItem] = Field(default_factory=list)
    recent_blood_pressure_labs: list[LabValueSummary] = Field(default_factory=list)
    current_antihypertensive_medications: list[MedicationMention] = Field(default_factory=list)

    # --- general clinical picture ---
    active_conditions: list[str] = Field(default_factory=list, description="Elixhauser (or equivalent) group names")
    chronic_conditions: list[str] = Field(default_factory=list)
    recent_labs: list[LabValueSummary] = Field(default_factory=list)
    current_medications: list[MedicationMention] = Field(default_factory=list)
    prior_medications: list[MedicationMention] = Field(default_factory=list)
    relevant_note_sections: list[NoteSection] = Field(default_factory=list)

    missing_information: list[MissingInformationFlag] = Field(default_factory=list)

    generated_at: datetime

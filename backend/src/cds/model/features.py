"""Phase 9: structured + text feature extraction from ClinicalState.

Every feature here reads only from `ClinicalState` (never raw MIMIC
DataFrames or discharge-time information) — this is what keeps the trained
model's input consistent with the leakage guards built into
`cds.timeline.state_builder` (Phase 5): `current_medications` is already
restricted to the early-admission window, and `relevant_note_sections`
already has the "Discharge Medications:" section truncated out.
"""
from typing import List

import pandas as pd

from ..schemas import ClinicalState

NUMERIC_FEATURES = [
    "age",
    "n_current_medications",
    "n_current_antihypertensive_medications",
    "lab_creatinine",
    "lab_potassium",
    "lab_sodium",
    "lab_bun",
]
CATEGORICAL_FEATURES = ["gender", "admission_reason", "hypertension_status"]
TEXT_FEATURE = "note_text"
# TfidfVectorizer raises ValueError("empty vocabulary") if every document in
# a fit is empty — a real risk here, since a meaningful fraction of
# admissions have zero usable note text (see Phase 0 audit's completeness
# numbers, and Phase 7's synthetic patients, which never populate notes at
# all). A non-empty placeholder keeps the vectorizer's vocabulary from ever
# being empty without meaningfully changing what it learns.
_NO_NOTES_PLACEHOLDER = "no_notes_available"
ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]

# Substring keywords (matched case-insensitively against LabValueSummary.label)
# used to pull a handful of hypertension/renal-safety-relevant labs into
# fixed-width numeric features. Not an exhaustive lab panel by design.
_LAB_FEATURE_KEYWORDS = {
    "lab_creatinine": ["creatinine"],
    "lab_potassium": ["potassium"],
    "lab_sodium": ["sodium"],
    "lab_bun": ["urea nitrogen", "bun"],
}


def _extract_lab_value(clinical_state: ClinicalState, keywords: List[str]) -> float:
    for lab in clinical_state.recent_labs:
        if lab.value is not None and any(k in lab.label.lower() for k in keywords):
            return lab.value
    return float("nan")


def build_structured_features(clinical_states: List[ClinicalState]) -> pd.DataFrame:
    """One row per ClinicalState, columns = ALL_FEATURE_COLUMNS (plus
    subject_id/hadm_id for traceability, not used as model input)."""
    rows = []
    for cs in clinical_states:
        row = {
            "subject_id": cs.subject_id,
            "hadm_id": cs.hadm_id,
            "age": cs.age,
            "gender": cs.gender.value,
            "admission_reason": cs.admission_reason,
            "hypertension_status": cs.hypertension_status.value,
            "n_current_medications": len(cs.current_medications),
            "n_current_antihypertensive_medications": len(cs.current_antihypertensive_medications),
            TEXT_FEATURE: " ".join(section.excerpt for section in cs.relevant_note_sections).strip()
            or _NO_NOTES_PLACEHOLDER,
        }
        for feature_name, keywords in _LAB_FEATURE_KEYWORDS.items():
            row[feature_name] = _extract_lab_value(cs, keywords)
        rows.append(row)
    return pd.DataFrame(rows, columns=["subject_id", "hadm_id"] + ALL_FEATURE_COLUMNS)

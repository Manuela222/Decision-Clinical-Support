"""Which of a patient's recent labs are relevant (and abnormal) for a
candidate medication class — Phase 12's `lookup_lab_abnormalities` MCP tool
wraps this."""
from typing import List

from ..schemas import ClinicalState, LabValueSummary
from .recommendation_safety import CREATININE_ABNORMAL_HIGH_MG_DL, RENAL_SENSITIVE_CLASSES

_RENAL_LAB_KEYWORDS = ["creatinine", "urea nitrogen", "bun", "potassium", "sodium"]


def lookup_lab_abnormalities(medication_class: str, clinical_state: ClinicalState) -> List[LabValueSummary]:
    """Return abnormal labs relevant to `medication_class`: for renal-
    sensitive classes, restricted to renal-function labs; otherwise any
    lab MIMIC (or this project) has flagged abnormal."""
    if medication_class in RENAL_SENSITIVE_CLASSES:
        candidates = [
            lab for lab in clinical_state.recent_labs
            if any(keyword in lab.label.lower() for keyword in _RENAL_LAB_KEYWORDS)
        ]
    else:
        candidates = clinical_state.recent_labs

    abnormal = []
    for lab in candidates:
        is_flagged = bool(lab.flag)
        is_creatinine_high = (
            "creatinine" in lab.label.lower() and lab.value is not None and lab.value > CREATININE_ABNORMAL_HIGH_MG_DL
        )
        if is_flagged or is_creatinine_high:
            abnormal.append(lab)
    return abnormal

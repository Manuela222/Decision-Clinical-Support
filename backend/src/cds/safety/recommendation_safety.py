"""Phase 10, part 1: deterministic, rule-based safety checks over a set of
MedicationRecommendations for one patient.

This is a pure function over already-built recommendations — it does not
generate or modify recommendations, only flags them. Baseline (Phase 8),
trained model (Phase 9), and agent (Phase 13) all call this the same way
and attach the result to `RecommendationResult.safety_warnings`.

The hypertension-unsafe check reuses each recommendation's own
`hypertension_compatible`/`hypertension_reasoning` (computed by
`cds.safety.check_hypertension_compatibility`) rather than re-deriving it,
so there is exactly one place that decision gets made.
"""
from collections import Counter
from typing import List, Optional, Tuple

from ..schemas import ClinicalState, LabValueSummary, MedicationRecommendation, SafetyCategory, SafetySeverity, SafetyWarning

# Renally-cleared or renal-function-sensitive medication classes: dosing or
# safety depends on kidney function, so an abnormal or missing creatinine is
# relevant regardless of how the recommendation was produced.
RENAL_SENSITIVE_CLASSES = {
    "nsaid",
    "ace inhibitor",
    "arb",
    "loop diuretic",
    "thiazide diuretic",
    "antidiabetic - biguanide",
    "antibiotic - aminoglycoside",
    "antiarrhythmic",
}

# mg/dL; a commonly used upper bound of normal for adult serum creatinine.
# A simplification (no age/sex-adjusted eGFR), appropriate for a rule-based
# prototype check, not a substitute for a real renal-dosing calculation.
CREATININE_ABNORMAL_HIGH_MG_DL = 1.3


def _find_creatinine(clinical_state: ClinicalState) -> Optional[LabValueSummary]:
    for lab in clinical_state.recent_labs:
        if "creatinine" in lab.label.lower():
            return lab
    return None


def check_renal_sensitivity(
    medication_class: str, clinical_state: ClinicalState
) -> Optional[Tuple[SafetySeverity, str, List[str]]]:
    """Return (severity, message, evidence_ids) if `medication_class` is
    renal-sensitive and this patient's renal labs are abnormal or missing;
    None if the class isn't renal-sensitive or renal function looks fine.

    Shared by `check_recommendation_safety` (below) and Phase 12's
    `check_medication_compatibility` MCP tool, so there is exactly one
    place this decision gets made.
    """
    if medication_class not in RENAL_SENSITIVE_CLASSES:
        return None
    creatinine_lab = _find_creatinine(clinical_state)
    if creatinine_lab is None:
        return (
            SafetySeverity.WARNING,
            f"'{medication_class}' is renal-sensitive but no recent creatinine value is available for this patient.",
            [],
        )
    if creatinine_lab.value is not None and creatinine_lab.value > CREATININE_ABNORMAL_HIGH_MG_DL:
        return (
            SafetySeverity.CRITICAL,
            f"'{medication_class}' is renal-sensitive and this patient's most recent creatinine "
            f"({creatinine_lab.value} {creatinine_lab.unit or ''}) is abnormal.",
            [creatinine_lab.evidence_id],
        )
    return None


def check_recommendation_safety(
    clinical_state: ClinicalState, recommendations: List[MedicationRecommendation]
) -> List[SafetyWarning]:
    warnings: List[SafetyWarning] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"safety-{clinical_state.hadm_id}-{counter}"

    # (a) duplicate medication classes
    class_counts = Counter(r.medication_class for r in recommendations)
    for medication_class, count in class_counts.items():
        if count > 1:
            warnings.append(
                SafetyWarning(
                    warning_id=next_id(),
                    severity=SafetySeverity.WARNING,
                    category=SafetyCategory.DUPLICATE_MEDICATION_CLASS,
                    message=f"'{medication_class}' is recommended {count} times.",
                    related_medication_class=medication_class,
                    evidence_ids=[],
                )
            )

    for rec in recommendations:
        # (b) hypertension-unsafe, regardless of how common the class is
        if not rec.hypertension_compatible:
            warnings.append(
                SafetyWarning(
                    warning_id=next_id(),
                    severity=SafetySeverity.CRITICAL,
                    category=SafetyCategory.HYPERTENSION_UNSAFE,
                    message=f"'{rec.medication_class}' flagged hypertension-unsafe: {rec.hypertension_reasoning}",
                    related_medication_class=rec.medication_class,
                    evidence_ids=list(rec.evidence_ids),
                )
            )

        # (c) renal-sensitive class + abnormal/missing renal labs
        renal_check = check_renal_sensitivity(rec.medication_class, clinical_state)
        if renal_check is not None:
            severity, message, evidence_ids = renal_check
            warnings.append(
                SafetyWarning(
                    warning_id=next_id(),
                    severity=severity,
                    category=SafetyCategory.RENAL_SENSITIVE,
                    message=message,
                    related_medication_class=rec.medication_class,
                    evidence_ids=evidence_ids,
                )
            )

        # (d) missing indication (no stated rationale for this recommendation)
        if not rec.rationale or not rec.rationale.strip():
            warnings.append(
                SafetyWarning(
                    warning_id=next_id(),
                    severity=SafetySeverity.WARNING,
                    category=SafetyCategory.MISSING_INDICATION,
                    message=f"No rationale/indication was given for recommending '{rec.medication_class}'.",
                    related_medication_class=rec.medication_class,
                    evidence_ids=[],
                )
            )

        # (e) missing evidence citations
        if not rec.evidence_ids:
            warnings.append(
                SafetyWarning(
                    warning_id=next_id(),
                    severity=SafetySeverity.INFO,
                    category=SafetyCategory.MISSING_EVIDENCE,
                    message=f"'{rec.medication_class}' has no cited evidence_ids backing this recommendation.",
                    related_medication_class=rec.medication_class,
                    evidence_ids=[],
                )
            )

    return warnings

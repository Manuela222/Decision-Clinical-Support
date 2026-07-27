"""Shared hypertension-compatibility primitive.

Every recommender (baseline, trained model, agent) must fill
`MedicationRecommendation.hypertension_compatible` /
`.hypertension_reasoning` for each candidate class — this is the one
function all three call to do it, so the verdict is consistent everywhere.

This intentionally covers only the hypertension-compatibility check.
Phase 10 builds the rest of the safety layer (duplicate classes, renal-
sensitivity, missing indication/evidence, drug interactions) as
`SafetyWarning`s around this same primitive — it does not duplicate this
logic, it imports it.
"""
from typing import Tuple

from ..schemas import HypertensionStatus

# Medication classes known to raise blood pressure or blunt antihypertensive
# efficacy, per the project spec's own examples (NSAIDs, decongestants).
# Class names must match cds.medications' fixed vocabulary exactly.
UNSAFE_FOR_HYPERTENSION = {
    "nsaid",
    "decongestant",
}

_HYPERTENSION_PRESENT_STATUSES = {HypertensionStatus.CONFIRMED_CHRONIC, HypertensionStatus.SUSPECTED}


def check_hypertension_compatibility(
    medication_class: str, hypertension_status: HypertensionStatus
) -> Tuple[bool, str]:
    """Return (is_compatible, human-readable reasoning) for one candidate
    medication class against the patient's hypertension status."""
    if hypertension_status not in _HYPERTENSION_PRESENT_STATUSES:
        return True, (
            f"Hypertension status is '{hypertension_status.value}' (not confirmed/suspected), "
            f"so no hypertension-specific restriction applies to '{medication_class}'."
        )
    if medication_class in UNSAFE_FOR_HYPERTENSION:
        return False, (
            f"'{medication_class}' can raise blood pressure or blunt antihypertensive control and is "
            f"flagged unsafe given the patient's {hypertension_status.value} hypertension."
        )
    return True, (
        f"'{medication_class}' has no known adverse interaction with hypertension management "
        f"for a patient with {hypertension_status.value} hypertension."
    )

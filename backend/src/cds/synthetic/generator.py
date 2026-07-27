"""Phase 7: fully synthetic patient generator.

*** SYNTHETIC DATA ONLY — NOT REAL MIMIC-III DATA ***
Every value produced by this module is drawn from fixed lists and random
ranges hardcoded below. Nothing here is sampled from, derived from, subset
of, or resembles any single real MIMIC-III patient row — there is no read
access to the real dataset anywhere in this file. Output is for pytest
fixtures, demos, and the "new patient" UI page (Phase 17) only, and must
never be presented as, or confused with, real patient data. See also the
top-level project README's "Synthetic data" section.

Subject/admission IDs are generated as negative integers specifically so
they can never collide with (or be mistaken for) a real MIMIC-III
SUBJECT_ID/HADM_ID, which are always positive.

Blood pressure is deliberately left empty here (not fabricated), to keep
this generator's output consistent with the real-patient pipeline, where BP
is structurally unavailable — see `cds.timeline.state_builder` module
docstring, point 3: BP lives in MIMIC-III's CHARTEVENTS table, which this
project does not load.
"""
import random
from datetime import datetime, timezone
from typing import List, Optional

from ..cohort.elixhauser import ALL_CATEGORIES
from ..medications import default_dictionary, is_antihypertensive_class
from ..schemas import (
    ClinicalState,
    Gender,
    HypertensionStatus,
    LabValueSummary,
    MedicationMention,
    MissingInformationFlag,
)

NON_HTN_ADMISSION_CONDITIONS = sorted(c for c in ALL_CATEGORIES if c != "Hypertension")

# (sentinel itemid, label, (low, high) plausible range, unit). Sentinel
# itemids are negative and are NOT real MIMIC-III D_LABITEMS.ITEMID values.
_SYNTHETIC_RENAL_LABS = [
    (-1, "Creatinine (synthetic)", (0.6, 1.3), "mg/dL"),
    (-2, "Blood urea nitrogen (synthetic)", (7, 20), "mg/dL"),
    (-3, "Potassium (synthetic)", (3.5, 5.1), "mEq/L"),
    (-4, "Sodium (synthetic)", (135, 145), "mEq/L"),
]


def generate_synthetic_clinical_state(
    seed: Optional[int] = None,
    hypertension_status: HypertensionStatus = HypertensionStatus.CONFIRMED_CHRONIC,
    subject_id: Optional[int] = None,
    hadm_id: Optional[int] = None,
) -> ClinicalState:
    """Generate one fully synthetic ClinicalState."""
    rng = random.Random(seed)
    evidence_counter = 0

    def next_evidence_id() -> str:
        nonlocal evidence_counter
        evidence_counter += 1
        return f"synthetic-{evidence_counter}"

    subject_id = subject_id if subject_id is not None else -rng.randint(1, 999_999)
    hadm_id = hadm_id if hadm_id is not None else -rng.randint(1, 999_999)

    age = rng.randint(18, 95)
    gender = rng.choice([Gender.M, Gender.F])
    admission_reason = rng.choice(NON_HTN_ADMISSION_CONDITIONS)

    active_conditions = [admission_reason]
    if hypertension_status == HypertensionStatus.CONFIRMED_CHRONIC:
        active_conditions.append("Hypertension")

    dictionary = default_dictionary()
    antihtn_names = sorted(n for n, cls in dictionary.classes.items() if is_antihypertensive_class(cls))
    other_names = sorted(n for n, cls in dictionary.classes.items() if not is_antihypertensive_class(cls))

    current_medications: List[MedicationMention] = []
    if hypertension_status in (HypertensionStatus.CONFIRMED_CHRONIC, HypertensionStatus.SUSPECTED) and antihtn_names:
        k = min(rng.randint(1, 2), len(antihtn_names))
        for name in rng.sample(antihtn_names, k=k):
            current_medications.append(
                MedicationMention(
                    normalized_name=name,
                    medication_class=dictionary.classes[name],
                    is_antihypertensive=True,
                    evidence_id=next_evidence_id(),
                )
            )
    k = min(rng.randint(1, 3), len(other_names))
    for name in rng.sample(other_names, k=k):
        current_medications.append(
            MedicationMention(
                normalized_name=name,
                medication_class=dictionary.classes[name],
                is_antihypertensive=False,
                evidence_id=next_evidence_id(),
            )
        )

    recent_labs: List[LabValueSummary] = [
        LabValueSummary(
            itemid=itemid,
            label=label,
            value=round(rng.uniform(lo, hi), 1),
            unit=unit,
            flag=None,
            charttime=None,
            evidence_id=next_evidence_id(),
        )
        for itemid, label, (lo, hi), unit in _SYNTHETIC_RENAL_LABS
    ]

    return ClinicalState(
        subject_id=subject_id,
        hadm_id=hadm_id,
        age=age,
        gender=gender,
        admission_reason=admission_reason,
        hypertension_status=hypertension_status,
        hypertension_evidence=[],
        recent_blood_pressure_labs=[],
        current_antihypertensive_medications=[m for m in current_medications if m.is_antihypertensive],
        active_conditions=active_conditions,
        chronic_conditions=active_conditions,
        recent_labs=recent_labs,
        current_medications=current_medications,
        prior_medications=[],
        relevant_note_sections=[],
        missing_information=[
            MissingInformationFlag.DISCHARGE_SUMMARY_MISSING,
            MissingInformationFlag.NOTES_MISSING,
            MissingInformationFlag.PRIOR_ADMISSIONS_MISSING,
            MissingInformationFlag.HYPERTENSION_LABS_MISSING,
        ],
        generated_at=datetime.now(timezone.utc),
    )


def generate_synthetic_cohort(n: int, seed: Optional[int] = None) -> List[ClinicalState]:
    """Generate `n` independent synthetic patients (distinct subject/hadm ids)."""
    rng = random.Random(seed)
    return [
        generate_synthetic_clinical_state(
            seed=rng.randint(0, 2**31 - 1),
            hypertension_status=rng.choice(
                [
                    HypertensionStatus.CONFIRMED_CHRONIC,
                    HypertensionStatus.CONFIRMED_CHRONIC,
                    HypertensionStatus.CONFIRMED_CHRONIC,
                    HypertensionStatus.SUSPECTED,
                ]
            ),
            subject_id=-(i + 1),
            hadm_id=-(i + 1),
        )
        for i in range(n)
    ]

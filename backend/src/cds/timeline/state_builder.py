"""Phase 5: build a compact, deterministic-heuristic ClinicalState from a
PatientTimeline. No LLM calls — every field is produced by fixed rules.

Two design decisions here are load-bearing for the rest of the project and
are documented inline rather than applied silently:

1. LEAKAGE GUARD: `current_medications` intentionally includes only
   medication orders started within `current_medications_window_hours` of
   admission (default 48h) — a proxy for "home medications / early hospital
   course". PRESCRIPTIONS has no explicit "home med" flag, so without this
   cutoff, `current_medications` would include the same late-stay orders
   Phase 4/8/9/14 use to define the *discharge medication* prediction
   target, handing the trained model and agent the answer as an input
   feature. Downstream phases must keep using this same ClinicalState
   object (not re-deriving "current medications" from raw PRESCRIPTIONS
   some other way) or this guard is silently defeated.

2. NOTE TEXT LEAKAGE GUARD: MIMIC-III discharge summaries routinely contain
   an explicit "Discharge Medications:" section listing the exact answer.
   `relevant_note_sections` truncates note text at the first occurrence of
   that (or a few equivalent) heading before excerpting, so note-derived
   features (Phase 9's text embeddings) can't trivially read off the label.
   This is a simple, auditable truncation, not NLP summarization.

3. DATA GAP: MIMIC-III blood pressure readings live in CHARTEVENTS (vital
   signs), which is *not* one of the eight tables this project loads (see
   Phase 2). `recent_blood_pressure_labs` is therefore always empty here,
   and `MissingInformationFlag.HYPERTENSION_LABS_MISSING` is set whenever
   hypertension is present/suspected — this is a real, structural data
   limitation, not a bug, and should be called out in the Phase 18 report.
"""
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from ..cohort.elixhauser import map_icd9_to_elixhauser
from ..medications import is_antihypertensive_class, map_to_medication_class, normalize_medication_name, EXCLUDED_CLASS
from ..schemas import (
    ClinicalState,
    Gender,
    HypertensionStatus,
    LabValueSummary,
    MedicationMention,
    MissingInformationFlag,
    NoteSection,
    PatientTimeline,
    TimelineEventType,
)

_DISCHARGE_MED_HEADING_RE = re.compile(
    r"(discharge medications?|medications on admission|discharge disposition)\s*:",
    re.IGNORECASE,
)
_MAX_NOTE_SECTIONS = 5
_NOTE_EXCERPT_CHARS = 800
_MAX_RECENT_LABS = 20


def _truncate_before_discharge_meds(text: str) -> str:
    match = _DISCHARGE_MED_HEADING_RE.search(text)
    return text[: match.start()] if match else text


def _medication_mention(normalized: str, evidence_id: str) -> Optional[MedicationMention]:
    med_class = map_to_medication_class(normalized)
    if med_class is None or med_class == EXCLUDED_CLASS:
        return None
    return MedicationMention(
        normalized_name=normalized,
        medication_class=med_class,
        is_antihypertensive=is_antihypertensive_class(med_class),
        evidence_id=evidence_id,
    )


def build_clinical_state(
    timeline: PatientTimeline,
    age: int,
    gender: str,
    condition_groups: List[str],
    admission_reason: str,
    current_medications_window_hours: int = 48,
    prior_admission_timeline: Optional[PatientTimeline] = None,
    generated_at: Optional[datetime] = None,
) -> ClinicalState:
    admission_events = [e for e in timeline.events if e.event_type == TimelineEventType.ADMISSION]
    admittime = admission_events[0].timestamp if admission_events else None
    window_cutoff = (
        admittime + timedelta(hours=current_medications_window_hours)
        if admittime is not None
        else None
    )

    diagnosis_events = [e for e in timeline.events if e.event_type == TimelineEventType.DIAGNOSIS]
    medication_events = [e for e in timeline.events if e.event_type == TimelineEventType.MEDICATION_ORDER]
    lab_events = [e for e in timeline.events if e.event_type == TimelineEventType.LAB_RESULT]
    note_events = [e for e in timeline.events if e.event_type == TimelineEventType.NOTE]
    discharge_summary_events = [e for e in note_events if e.details.get("category", "").strip() == "Discharge summary"]

    hypertension_evidence = [
        e.evidence
        for e in diagnosis_events
        if "Hypertension" in map_icd9_to_elixhauser(e.details.get("icd9_code", ""))
    ]

    current_medications: List[MedicationMention] = []
    for e in medication_events:
        if window_cutoff is not None and e.timestamp is not None and e.timestamp > window_cutoff:
            continue
        normalized = normalize_medication_name(e.details.get("drug", ""))
        mention = _medication_mention(normalized, e.evidence.evidence_id)
        if mention is not None:
            current_medications.append(mention)

    prior_medications: List[MedicationMention] = []
    if prior_admission_timeline is not None:
        for e in prior_admission_timeline.events:
            if e.event_type != TimelineEventType.MEDICATION_ORDER:
                continue
            normalized = normalize_medication_name(e.details.get("drug", ""))
            mention = _medication_mention(normalized, e.evidence.evidence_id)
            if mention is not None:
                prior_medications.append(mention)

    current_antihypertensive_medications = [m for m in current_medications if m.is_antihypertensive]

    if "Hypertension" in condition_groups:
        hypertension_status = HypertensionStatus.CONFIRMED_CHRONIC
    elif current_antihypertensive_medications:
        hypertension_status = HypertensionStatus.SUSPECTED
    elif condition_groups or current_medications:
        hypertension_status = HypertensionStatus.NOT_PRESENT
    else:
        hypertension_status = HypertensionStatus.UNKNOWN

    recent_labs: List[LabValueSummary] = []
    sorted_labs = sorted(lab_events, key=lambda e: (e.timestamp is None, e.timestamp), reverse=True)
    for e in sorted_labs[:_MAX_RECENT_LABS]:
        recent_labs.append(
            LabValueSummary(
                itemid=e.details["itemid"],
                label=e.details["label"],
                value=e.details.get("valuenum"),
                unit=e.details.get("valueuom") or None,
                flag=e.details.get("flag"),
                charttime=e.timestamp,
                evidence_id=e.evidence.evidence_id,
            )
        )

    relevant_note_sections: List[NoteSection] = []
    sorted_notes = sorted(note_events, key=lambda e: (e.timestamp is None, e.timestamp), reverse=True)
    for e in sorted_notes[:_MAX_NOTE_SECTIONS]:
        safe_text = _truncate_before_discharge_meds(e.details.get("text", ""))
        excerpt = safe_text.strip()[:_NOTE_EXCERPT_CHARS]
        if not excerpt:
            continue
        relevant_note_sections.append(
            NoteSection(
                category=e.details.get("category", ""),
                section_title=None,
                excerpt=excerpt,
                evidence_id=e.evidence.evidence_id,
            )
        )

    missing_information: List[MissingInformationFlag] = []
    if not discharge_summary_events:
        missing_information.append(MissingInformationFlag.DISCHARGE_SUMMARY_MISSING)
    if not lab_events:
        missing_information.append(MissingInformationFlag.LABS_MISSING)
    if not note_events:
        missing_information.append(MissingInformationFlag.NOTES_MISSING)
    if prior_admission_timeline is None:
        missing_information.append(MissingInformationFlag.PRIOR_ADMISSIONS_MISSING)
    # Always true given this project's data sources (see module docstring,
    # point 3): blood pressure lives in CHARTEVENTS, which isn't loaded.
    missing_information.append(MissingInformationFlag.HYPERTENSION_LABS_MISSING)

    return ClinicalState(
        subject_id=timeline.subject_id,
        hadm_id=timeline.hadm_id,
        age=age,
        gender=Gender(gender),
        admission_reason=admission_reason,
        hypertension_status=hypertension_status,
        hypertension_evidence=hypertension_evidence,
        recent_blood_pressure_labs=[],
        current_antihypertensive_medications=current_antihypertensive_medications,
        active_conditions=list(condition_groups),
        chronic_conditions=list(condition_groups),
        recent_labs=recent_labs,
        current_medications=current_medications,
        prior_medications=prior_medications,
        relevant_note_sections=relevant_note_sections,
        missing_information=missing_information,
        generated_at=generated_at or datetime.now(timezone.utc),
    )

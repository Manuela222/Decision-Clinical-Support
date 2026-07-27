"""Phase 11: focused, evidence-cited retrieval functions over a
PatientTimeline. Phase 12's MCP tools wrap these directly.

*** LEAKAGE WARNING ***
`search_notes`, `get_recent_labs`, `get_medications`, and `get_diagnoses`
return EVERY matching event in the given timeline, regardless of date —
including events at/after discharge. That's the raw, unfiltered record
(Phase 5's PatientTimeline), not the leakage-guarded ClinicalState. Only
ever call these four with a TRAIN-split patient's timeline (e.g. one of
`find_similar_patient_profiles`'s results) — never with the query/test
patient's own current-admission timeline, or you hand over the discharge
medication answer through a side channel that bypasses ClinicalState's
48-hour medication-window guard entirely.

`get_evidence_by_ids` is safe to use on any timeline (query patient
included): it only returns evidence for IDs the caller explicitly already
has (typically ones already surfaced via ClinicalState), never a bulk dump.
"""
from typing import List

from ..schemas import EvidenceItem, PatientTimeline, TimelineEventType


def get_evidence_by_ids(timeline: PatientTimeline, evidence_ids: List[str]) -> List[EvidenceItem]:
    by_id = {event.evidence.evidence_id: event.evidence for event in timeline.events}
    return [by_id[eid] for eid in evidence_ids if eid in by_id]


def search_notes(timeline: PatientTimeline, keyword: str) -> List[EvidenceItem]:
    """Case-insensitive substring search over NOTE event text. See module
    docstring's leakage warning — train-split timelines only."""
    keyword_lower = keyword.lower()
    return [
        event.evidence
        for event in timeline.events
        if event.event_type == TimelineEventType.NOTE and keyword_lower in event.details.get("text", "").lower()
    ]


def get_recent_labs(timeline: PatientTimeline, limit: int = 20) -> List[EvidenceItem]:
    """Most recent lab results first. See module docstring's leakage warning."""
    lab_events = sorted(
        (event for event in timeline.events if event.event_type == TimelineEventType.LAB_RESULT),
        key=lambda event: (event.timestamp is None, event.timestamp),
        reverse=True,
    )
    return [event.evidence for event in lab_events[:limit]]


def get_medications(timeline: PatientTimeline) -> List[EvidenceItem]:
    """Every medication order in the timeline. See module docstring's leakage warning."""
    return [event.evidence for event in timeline.events if event.event_type == TimelineEventType.MEDICATION_ORDER]


def get_diagnoses(timeline: PatientTimeline) -> List[EvidenceItem]:
    """Every diagnosis code in the timeline. See module docstring's leakage warning."""
    return [event.evidence for event in timeline.events if event.event_type == TimelineEventType.DIAGNOSIS]

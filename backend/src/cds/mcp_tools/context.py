"""Server-side context for one patient session's fixed tool set.

Holds data the agent (Phase 13) never sees directly in its own context —
the raw PatientTimeline and the train-split RAG index. The agent only ever
sees what a tool call explicitly returns for the specific arguments it
passed, never a bulk dump of either.

`patient_timeline` is the QUERY patient's own full timeline. It exists here
ONLY so `get_evidence_citations` can resolve an evidence_id already
surfaced via ClinicalState back to its full source-row provenance. Per the
leakage warning in `cds.retrieval.timeline_queries`, nothing in this tool
layer calls `search_notes`/`get_recent_labs`/`get_medications`/
`get_diagnoses` against this timeline — that would bypass ClinicalState's
medication-window leakage guard entirely. Those four retrieval functions
are for drilling into a *train-split similar patient's* timeline, not the
query patient's own.
"""
from dataclasses import dataclass

from ..retrieval import PatientProfileIndex
from ..schemas import ClinicalState, PatientTimeline


@dataclass
class MCPToolContext:
    clinical_state: ClinicalState
    patient_timeline: PatientTimeline
    patient_profile_index: PatientProfileIndex

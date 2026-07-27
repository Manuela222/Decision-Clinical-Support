"""Patient timeline: raw chronological facts, no summarization, no recommendations."""
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .common import EvidenceItem


class TimelineEventType(str, Enum):
    ADMISSION = "admission"
    DISCHARGE = "discharge"
    DIAGNOSIS = "diagnosis"
    MEDICATION_ORDER = "medication_order"
    LAB_RESULT = "lab_result"
    NOTE = "note"


class TimelineEvent(BaseModel):
    """A single dated fact about a patient. No interpretation applied here."""

    event_id: str
    subject_id: int
    hadm_id: Optional[int] = None
    event_type: TimelineEventType
    timestamp: Optional[datetime] = Field(
        default=None, description="None only when the source row itself has no timestamp"
    )
    description: str
    details: dict[str, Any] = Field(
        default_factory=dict, description="JSON-serializable structured payload specific to event_type"
    )
    evidence: EvidenceItem


class PatientTimeline(BaseModel):
    """Chronological, deterministic timeline for one admission. No LLM involvement."""

    subject_id: int
    hadm_id: int
    events: list[TimelineEvent] = Field(default_factory=list)
    generated_at: datetime

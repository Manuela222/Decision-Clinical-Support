"""Shared primitives: evidence citations and the source-table vocabulary.

Every downstream claim (timeline event, clinical-state field, recommendation
rationale, safety warning) must be traceable back to a source row via an
EvidenceItem — this is the project's traceability/XAI requirement.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceTable(str, Enum):
    PATIENTS = "PATIENTS"
    ADMISSIONS = "ADMISSIONS"
    DIAGNOSES_ICD = "DIAGNOSES_ICD"
    PRESCRIPTIONS = "PRESCRIPTIONS"
    NOTEEVENTS = "NOTEEVENTS"
    LABEVENTS = "LABEVENTS"
    D_LABITEMS = "D_LABITEMS"
    D_ICD_DIAGNOSES = "D_ICD_DIAGNOSES"
    DERIVED = "DERIVED"  # for evidence computed by our own pipeline, not a raw MIMIC row


class EvidenceItem(BaseModel):
    """A single, traceable piece of evidence backing a claim or recommendation."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(..., description="Stable, unique identifier for this evidence item")
    source_table: SourceTable
    source_row_id: Optional[str] = Field(
        default=None, description="ROW_ID (or composite key) of the source row, as a string"
    )
    subject_id: int
    hadm_id: Optional[int] = None
    description: str = Field(..., description="Human-readable summary of what this evidence shows")
    value: Optional[str] = Field(default=None, description="Raw or normalized value, if applicable")
    timestamp: Optional[datetime] = Field(default=None, description="Event/observation time, if applicable")
    text_excerpt: Optional[str] = Field(
        default=None, description="Short excerpt of source text, e.g. a note snippet"
    )

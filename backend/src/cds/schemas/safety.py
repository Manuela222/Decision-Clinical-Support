"""Deterministic, rule-based safety warnings (Phase 10)."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SafetySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SafetyCategory(str, Enum):
    DUPLICATE_MEDICATION_CLASS = "duplicate_medication_class"
    HYPERTENSION_UNSAFE = "hypertension_unsafe"
    RENAL_SENSITIVE = "renal_sensitive"
    MISSING_INDICATION = "missing_indication"
    MISSING_EVIDENCE = "missing_evidence"
    DRUG_INTERACTION = "drug_interaction"


class SafetyWarning(BaseModel):
    warning_id: str
    severity: SafetySeverity
    category: SafetyCategory
    message: str
    related_medication_class: Optional[str] = None
    evidence_ids: list[str] = Field(default_factory=list)

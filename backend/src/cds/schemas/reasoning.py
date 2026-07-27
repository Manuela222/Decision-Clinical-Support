"""Traceable reasoning steps for the agentic recommender (Phase 13).

No hidden chain-of-thought: each step is a concise, clinician-facing summary
of a retrieval, tool call, or reasoning conclusion — never raw model
scratch-space text.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ReasoningStepType(str, Enum):
    RETRIEVAL = "retrieval"
    TOOL_CALL = "tool_call"
    HYPERTENSION_COMPATIBILITY_CHECK = "hypertension_compatibility_check"
    REASONING = "reasoning"
    CONCLUSION = "conclusion"


class ReasoningStep(BaseModel):
    step_index: int
    step_type: ReasoningStepType
    description: str = Field(..., description="Concise, clinician-facing summary — not raw chain-of-thought")
    tool_name: Optional[str] = None
    tool_input: Optional[dict[str, Any]] = None
    tool_output: Optional[dict[str, Any]] = None
    evidence_ids: list[str] = Field(default_factory=list)
    timestamp: datetime

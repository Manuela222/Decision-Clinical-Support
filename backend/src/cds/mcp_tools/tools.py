"""Phase 12: the 5 fixed MCP tool implementations, as plain Python
functions over an MCPToolContext. This is the business-logic layer —
server.py is a thin FastMCP adapter around exactly these functions, and
tests call these directly ("fake tool calls") without needing a live
MCP transport.
"""
from typing import List

from ..retrieval import SimilarPatientProfile, find_similar_patient_profiles, get_evidence_by_ids
from ..safety import check_hypertension_compatibility, check_renal_sensitivity
from ..safety import check_drug_interactions as _check_drug_interactions
from ..safety import lookup_lab_abnormalities as _lookup_lab_abnormalities
from ..safety.drug_interactions import DrugInteractionWarning
from ..schemas import EvidenceItem, LabValueSummary
from .context import MCPToolContext
from .schemas import MedicationCompatibilityResult, RenalConcern


def search_similar_patient_profiles(ctx: MCPToolContext, top_k: int = 5) -> List[SimilarPatientProfile]:
    """Tool 1: RAG lookup over train-split ClinicalStates similar to this
    patient, each with its actual discharge medication classes."""
    return find_similar_patient_profiles(ctx.clinical_state, ctx.patient_profile_index, top_k=top_k)


def check_medication_compatibility(ctx: MCPToolContext, medication_class: str) -> MedicationCompatibilityResult:
    """Tool 2: hypertension-compatibility + renal-sensitivity check for one
    candidate medication class against this patient's structured state."""
    compatible, reasoning = check_hypertension_compatibility(medication_class, ctx.clinical_state.hypertension_status)
    renal_check = check_renal_sensitivity(medication_class, ctx.clinical_state)
    renal_concern = (
        RenalConcern(severity=renal_check[0], message=renal_check[1], evidence_ids=renal_check[2])
        if renal_check is not None
        else None
    )
    return MedicationCompatibilityResult(
        medication_class=medication_class,
        hypertension_compatible=compatible,
        hypertension_reasoning=reasoning,
        renal_concern=renal_concern,
    )


def lookup_lab_abnormalities(ctx: MCPToolContext, medication_class: str) -> List[LabValueSummary]:
    """Tool 3: flagged abnormal labs relevant to a candidate medication."""
    return _lookup_lab_abnormalities(medication_class, ctx.clinical_state)


def check_drug_interactions(ctx: MCPToolContext, candidate_medication_class: str) -> List[DrugInteractionWarning]:
    """Tool 4: rule-based interaction check against the patient's current medications."""
    return _check_drug_interactions(candidate_medication_class, ctx.clinical_state)


def get_evidence_citations(ctx: MCPToolContext, evidence_ids: List[str]) -> List[EvidenceItem]:
    """Tool 5: resolve evidence_ids already surfaced via ClinicalState back
    to their full source-row provenance, for traceability."""
    return get_evidence_by_ids(ctx.patient_timeline, evidence_ids)

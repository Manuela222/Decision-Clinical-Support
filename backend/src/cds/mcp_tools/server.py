"""Phase 12: MCP server exposing exactly 5 fixed tools — no dynamically
invented tools. Thin adapter only: every tool body is a one-line call into
tools.py (the actual business logic), which is what tests exercise directly.

Because the tool set is scoped to one patient (an MCPToolContext bound at
construction), `build_mcp_server` takes that context and returns a server
configured for that single session — matching how Phase 13's agent is
expected to interact with one patient at a time.
"""
from typing import List

from mcp.server.fastmcp import FastMCP

from ..retrieval import SimilarPatientProfile
from ..safety.drug_interactions import DrugInteractionWarning
from ..schemas import EvidenceItem, LabValueSummary
from . import tools
from .context import MCPToolContext
from .schemas import MedicationCompatibilityResult

FIXED_TOOL_NAMES = [
    "search_similar_patient_profiles",
    "check_medication_compatibility",
    "lookup_lab_abnormalities",
    "check_drug_interactions",
    "get_evidence_citations",
]


def build_mcp_server(ctx: MCPToolContext, name: str = "cds-clinical-tools") -> FastMCP:
    """Build a FastMCP server with exactly the 5 fixed tools, bound to one
    patient's MCPToolContext."""
    server = FastMCP(name=name)

    @server.tool(name="search_similar_patient_profiles")
    def _search_similar_patient_profiles(top_k: int = 5) -> List[SimilarPatientProfile]:
        """RAG lookup over train-split ClinicalStates similar to this patient,
        each with its actual discharge medication classes."""
        return tools.search_similar_patient_profiles(ctx, top_k=top_k)

    @server.tool(name="check_medication_compatibility")
    def _check_medication_compatibility(medication_class: str) -> MedicationCompatibilityResult:
        """Check a candidate medication class against this patient's
        hypertension status and renal-function labs."""
        return tools.check_medication_compatibility(ctx, medication_class=medication_class)

    @server.tool(name="lookup_lab_abnormalities")
    def _lookup_lab_abnormalities(medication_class: str) -> List[LabValueSummary]:
        """Return flagged abnormal labs relevant to a candidate medication class."""
        return tools.lookup_lab_abnormalities(ctx, medication_class=medication_class)

    @server.tool(name="check_drug_interactions")
    def _check_drug_interactions(candidate_medication_class: str) -> List[DrugInteractionWarning]:
        """Local rule-based interaction check against the patient's current medications."""
        return tools.check_drug_interactions(ctx, candidate_medication_class=candidate_medication_class)

    @server.tool(name="get_evidence_citations")
    def _get_evidence_citations(evidence_ids: List[str]) -> List[EvidenceItem]:
        """Resolve evidence_ids to their full EvidenceItem source-row provenance."""
        return tools.get_evidence_citations(ctx, evidence_ids=evidence_ids)

    return server

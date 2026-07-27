"""Typed input/output schemas for the 5 fixed MCP tools (Phase 12).

Documented per tool below (also see server.py's docstrings, which FastMCP
surfaces to the client as the tool descriptions):

1. search_similar_patient_profiles
   in:  SearchSimilarPatientProfilesInput { top_k: int = 5 }
   out: list[SimilarPatientProfile]  (cds.retrieval.SimilarPatientProfile)

2. check_medication_compatibility
   in:  CheckMedicationCompatibilityInput { medication_class: str }
   out: MedicationCompatibilityResult

3. lookup_lab_abnormalities
   in:  LookupLabAbnormalitiesInput { medication_class: str }
   out: list[LabValueSummary]  (cds.schemas.LabValueSummary)

4. check_drug_interactions
   in:  CheckDrugInteractionsInput { candidate_medication_class: str }
   out: list[DrugInteractionWarning]  (cds.safety.DrugInteractionWarning)

5. get_evidence_citations
   in:  GetEvidenceCitationsInput { evidence_ids: list[str] }
   out: list[EvidenceItem]  (cds.schemas.EvidenceItem)
"""
from typing import List, Optional

from pydantic import BaseModel

from ..schemas import SafetySeverity


class SearchSimilarPatientProfilesInput(BaseModel):
    top_k: int = 5


class CheckMedicationCompatibilityInput(BaseModel):
    medication_class: str


class RenalConcern(BaseModel):
    severity: SafetySeverity
    message: str
    evidence_ids: List[str]


class MedicationCompatibilityResult(BaseModel):
    medication_class: str
    hypertension_compatible: bool
    hypertension_reasoning: str
    renal_concern: Optional[RenalConcern] = None


class LookupLabAbnormalitiesInput(BaseModel):
    medication_class: str


class CheckDrugInteractionsInput(BaseModel):
    candidate_medication_class: str


class GetEvidenceCitationsInput(BaseModel):
    evidence_ids: List[str]

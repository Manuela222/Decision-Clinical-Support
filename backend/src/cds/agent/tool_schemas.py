"""OpenAI-format tool schemas for the fixed Phase 12 tool list, built
directly from the same Pydantic input models `cds.mcp_tools.server` uses —
one definition of each tool's input shape, not two."""
from typing import Any, Dict, List

from ..mcp_tools.schemas import (
    CheckDrugInteractionsInput,
    CheckMedicationCompatibilityInput,
    GetEvidenceCitationsInput,
    LookupLabAbnormalitiesInput,
    SearchSimilarPatientProfilesInput,
)

_TOOL_INPUT_MODELS = {
    "search_similar_patient_profiles": (
        SearchSimilarPatientProfilesInput,
        "RAG lookup over train-split ClinicalStates similar to this patient, each with its actual "
        "discharge medication classes.",
    ),
    "check_medication_compatibility": (
        CheckMedicationCompatibilityInput,
        "Check a candidate medication class against this patient's hypertension status and renal-function labs.",
    ),
    "lookup_lab_abnormalities": (
        LookupLabAbnormalitiesInput,
        "Return flagged abnormal labs relevant to a candidate medication class.",
    ),
    "check_drug_interactions": (
        CheckDrugInteractionsInput,
        "Local rule-based interaction check against the patient's current medications.",
    ),
    "get_evidence_citations": (
        GetEvidenceCitationsInput,
        "Resolve evidence_ids to their full EvidenceItem source-row provenance.",
    ),
}


def build_openai_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": model_cls.model_json_schema(),
            },
        }
        for name, (model_cls, description) in _TOOL_INPUT_MODELS.items()
    ]

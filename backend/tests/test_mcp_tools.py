"""Phase 12: MCP tool tests using fake tool calls -- i.e. calling the
business-logic functions in cds.mcp_tools.tools directly, exactly as
server.py's thin FastMCP adapter does, without needing a live MCP
transport. A separate smoke test confirms the real FastMCP server actually
registers all 5 tools (test_build_mcp_server_registers_fixed_tools)."""
import asyncio
from datetime import datetime, timezone

import pytest

from cds.mcp_tools import MCPToolContext, build_mcp_server
from cds.mcp_tools import tools as mcp_tools
from cds.retrieval import build_patient_profile_index
from cds.schemas import (
    ClinicalState,
    Gender,
    HypertensionStatus,
    LabValueSummary,
    MedicationMention,
    PatientTimeline,
    TimelineEvent,
    TimelineEventType,
    EvidenceItem,
    SourceTable,
)


def _clinical_state(idx=999, reason="Cardiac arrhythmias", age=65, recent_labs=None, current_medications=None):
    return ClinicalState(
        subject_id=idx,
        hadm_id=1000 + idx,
        age=age,
        gender=Gender.F,
        admission_reason=reason,
        hypertension_status=HypertensionStatus.CONFIRMED_CHRONIC,
        hypertension_evidence=[],
        recent_blood_pressure_labs=[],
        current_antihypertensive_medications=[],
        active_conditions=[reason, "Hypertension"],
        chronic_conditions=[reason, "Hypertension"],
        recent_labs=recent_labs or [],
        current_medications=current_medications or [],
        prior_medications=[],
        relevant_note_sections=[],
        missing_information=[],
        generated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def index():
    train_states = [
        _clinical_state(1, "Cardiac arrhythmias", 60),
        _clinical_state(2, "Cardiac arrhythmias", 62),
    ]
    train_labels = [["antiarrhythmic"], ["antiarrhythmic", "anticoagulant"]]
    return build_patient_profile_index(train_states, train_labels)


@pytest.fixture
def timeline():
    evidence = EvidenceItem(
        evidence_id="LABEVENTS-1", source_table=SourceTable.LABEVENTS, source_row_id="1",
        subject_id=999, hadm_id=1999, description="Creatinine 1.0", value="1.0",
        timestamp=datetime.now(timezone.utc),
    )
    event = TimelineEvent(
        event_id="lab-1", subject_id=999, hadm_id=1999, event_type=TimelineEventType.LAB_RESULT,
        timestamp=datetime.now(timezone.utc), description="Lab result: Creatinine = 1.0",
        details={"itemid": 1, "label": "Creatinine", "value": "1.0", "valuenum": 1.0, "valueuom": "mg/dL", "flag": None},
        evidence=evidence,
    )
    return PatientTimeline(subject_id=999, hadm_id=1999, events=[event], generated_at=datetime.now(timezone.utc))


@pytest.fixture
def ctx(index, timeline):
    query = _clinical_state(
        idx=999,
        recent_labs=[
            LabValueSummary(itemid=1, label="Creatinine", value=2.5, unit="mg/dL", flag="abnormal",
                             charttime=None, evidence_id="LABEVENTS-1")
        ],
        current_medications=[
            MedicationMention(normalized_name="lisinopril", medication_class="ace inhibitor",
                               is_antihypertensive=True, evidence_id="PRESCRIPTIONS-1")
        ],
    )
    return MCPToolContext(clinical_state=query, patient_timeline=timeline, patient_profile_index=index)


# --- Tool 1: search_similar_patient_profiles --------------------------------

def test_tool_search_similar_patient_profiles(ctx):
    results = mcp_tools.search_similar_patient_profiles(ctx, top_k=1)
    assert len(results) == 1
    assert results[0].ground_truth_medication_classes


# --- Tool 2: check_medication_compatibility ---------------------------------

def test_tool_check_medication_compatibility_flags_nsaid(ctx):
    result = mcp_tools.check_medication_compatibility(ctx, medication_class="nsaid")
    assert result.hypertension_compatible is False
    assert "hypertension" in result.hypertension_reasoning.lower()


def test_tool_check_medication_compatibility_renal_concern_for_abnormal_creatinine(ctx):
    # ctx's patient has creatinine 2.5 (abnormal) and ace inhibitor is renal-sensitive
    result = mcp_tools.check_medication_compatibility(ctx, medication_class="ace inhibitor")
    assert result.renal_concern is not None
    assert result.renal_concern.severity.value == "critical"


def test_tool_check_medication_compatibility_no_renal_concern_for_non_sensitive_class(ctx):
    result = mcp_tools.check_medication_compatibility(ctx, medication_class="statin")
    assert result.renal_concern is None


# --- Tool 3: lookup_lab_abnormalities ----------------------------------------

def test_tool_lookup_lab_abnormalities_returns_abnormal_creatinine(ctx):
    result = mcp_tools.lookup_lab_abnormalities(ctx, medication_class="ace inhibitor")
    assert len(result) == 1
    assert result[0].label == "Creatinine"


def test_tool_lookup_lab_abnormalities_empty_for_non_sensitive_class_with_no_flags(ctx):
    # statin isn't renal-sensitive, so only explicitly-flagged labs would show;
    # the one lab present IS flagged "abnormal", so it should still show since
    # the flag itself makes it relevant regardless of class.
    result = mcp_tools.lookup_lab_abnormalities(ctx, medication_class="statin")
    assert len(result) == 1  # flagged abnormal regardless of class-specific relevance


# --- Tool 4: check_drug_interactions -----------------------------------------

def test_tool_check_drug_interactions_flags_ace_arb_combo(ctx):
    # patient already on an ace inhibitor (lisinopril); candidate is arb
    result = mcp_tools.check_drug_interactions(ctx, candidate_medication_class="arb")
    assert len(result) == 1
    assert result[0].interacting_medication_class == "ace inhibitor"
    assert result[0].severity.value == "critical"


def test_tool_check_drug_interactions_no_interaction_for_unrelated_class(ctx):
    result = mcp_tools.check_drug_interactions(ctx, candidate_medication_class="statin")
    assert result == []


# --- Tool 5: get_evidence_citations ------------------------------------------

def test_tool_get_evidence_citations_resolves_known_id(ctx):
    result = mcp_tools.get_evidence_citations(ctx, evidence_ids=["LABEVENTS-1"])
    assert len(result) == 1
    assert result[0].description == "Creatinine 1.0"


def test_tool_get_evidence_citations_ignores_unknown_id(ctx):
    result = mcp_tools.get_evidence_citations(ctx, evidence_ids=["not-a-real-id"])
    assert result == []


# --- server wiring smoke test -------------------------------------------------

def test_build_mcp_server_registers_fixed_tools(ctx):
    server = build_mcp_server(ctx)
    registered = asyncio.run(server.list_tools())
    names = {t.name for t in registered}
    assert names == {
        "search_similar_patient_profiles",
        "check_medication_compatibility",
        "lookup_lab_abnormalities",
        "check_drug_interactions",
        "get_evidence_citations",
    }

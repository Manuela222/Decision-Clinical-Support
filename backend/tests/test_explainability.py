"""Phase 15: explainability report tests."""
import json
from datetime import datetime, timezone

import pytest

from cds.agent import LLMTurn, MockLLMProvider, ToolCallRequest, recommend_agent
from cds.baseline import compute_diagnosis_medication_stats, recommend_baseline
from cds.explainability import explain_agent, explain_baseline, explain_trained_model
from cds.mcp_tools import MCPToolContext
from cds.model import recommend_trained_model, train_model
from cds.retrieval import build_patient_profile_index
from cds.schemas import ClinicalState, Gender, HypertensionStatus, LabValueSummary, PatientTimeline


def _clinical_state(idx=999, reason="Cardiac arrhythmias", age=65, creatinine=1.0):
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
        recent_labs=[
            LabValueSummary(itemid=1, label="Creatinine", value=creatinine, unit="mg/dL", flag=None,
                             charttime=None, evidence_id=f"ev-{idx}")
        ],
        current_medications=[],
        prior_medications=[],
        relevant_note_sections=[],
        missing_information=[],
        generated_at=datetime.now(timezone.utc),
    )


# --- explain_baseline ---------------------------------------------------------

def test_explain_baseline_reports_accepted_and_rejected_candidates():
    reasons = ["Cardiac arrhythmias"] * 4
    classes = [
        ["antiarrhythmic", "statin"],
        ["antiarrhythmic"],
        ["antiarrhythmic"],
        ["statin"],
    ]
    stats = compute_diagnosis_medication_stats(reasons, classes)

    state = _clinical_state()
    result = recommend_baseline(state, stats, min_frequency=0.6)
    report = explain_baseline(state, result, stats, ground_truth_medications=["antiarrhythmic"])

    assert report.method.value == "baseline"
    accepted = {c.medication_class for c in report.medication_candidates if c.accepted}
    rejected = {c.medication_class for c in report.medication_candidates if not c.accepted}
    assert accepted == {"antiarrhythmic"}
    assert "statin" in rejected
    assert report.matched_classes == ["antiarrhythmic"]
    assert report.missed_classes == []
    assert report.extra_classes == []


# --- explain_trained_model -----------------------------------------------------

def _train_artifact():
    states = [_clinical_state(i, "Cardiac arrhythmias" if i % 2 == 0 else "Diabetes, uncomplicated", 40 + i, 0.6 + 0.01 * i) for i in range(30)]
    labels = [["antiarrhythmic"] if i % 2 == 0 else ["antidiabetic - biguanide"] for i in range(30)]
    return train_model(states, labels, seed=1)


def test_explain_trained_model_includes_feature_importances_and_all_candidates():
    artifact = _train_artifact()
    query = _clinical_state()
    result = recommend_trained_model(query, artifact, threshold=0.0)
    report = explain_trained_model(query, result, artifact, ground_truth_medications=["antiarrhythmic"])

    assert len(report.feature_importances) > 0
    assert sum(f.importance for f in report.feature_importances) >= 0
    candidate_classes = {c.medication_class for c in report.medication_candidates}
    assert candidate_classes == set(artifact.label_binarizer.classes_)
    for candidate in report.medication_candidates:
        assert candidate.score is not None


# --- explain_agent --------------------------------------------------------------

@pytest.fixture
def agent_ctx():
    train_states = [_clinical_state(1, "Cardiac arrhythmias"), _clinical_state(2, "Cardiac arrhythmias")]
    train_labels = [["antiarrhythmic"], ["antiarrhythmic", "anticoagulant"]]
    index = build_patient_profile_index(train_states, train_labels)
    timeline = PatientTimeline(subject_id=999, hadm_id=1999, events=[], generated_at=datetime.now(timezone.utc))
    return MCPToolContext(clinical_state=_clinical_state(), patient_timeline=timeline, patient_profile_index=index)


def test_explain_agent_captures_retrieved_profiles_and_considered_candidates(agent_ctx):
    provider = MockLLMProvider(
        [
            LLMTurn(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="search_similar_patient_profiles", arguments={"top_k": 2})],
            ),
            LLMTurn(
                content=None,
                tool_calls=[ToolCallRequest(id="call_2", name="check_medication_compatibility", arguments={"medication_class": "nsaid"})],
            ),
            LLMTurn(
                content=json.dumps(
                    {"recommendations": [{"medication_class": "antiarrhythmic", "action": "start", "rationale": "Rate control.", "confidence": 0.8}]}
                ),
                tool_calls=[],
            ),
        ]
    )
    result = recommend_agent(agent_ctx.clinical_state, agent_ctx, provider)
    report = explain_agent(agent_ctx.clinical_state, result, ground_truth_medications=["antiarrhythmic"])

    assert len(report.retrieved_similar_profiles) == 2
    candidate_classes = {c.medication_class for c in report.medication_candidates}
    assert "antiarrhythmic" in candidate_classes  # accepted
    assert "nsaid" in candidate_classes  # considered (compatibility checked) but rejected
    nsaid_candidate = next(c for c in report.medication_candidates if c.medication_class == "nsaid")
    assert nsaid_candidate.accepted is False
    assert report.matched_classes == ["antiarrhythmic"]


# --- serialization ---------------------------------------------------------------

def test_explainability_report_round_trips_json():
    reasons = ["Cardiac arrhythmias"]
    classes = [["antiarrhythmic"]]
    stats = compute_diagnosis_medication_stats(reasons, classes)
    state = _clinical_state()
    result = recommend_baseline(state, stats, min_frequency=0.0)
    report = explain_baseline(state, result, stats, ground_truth_medications=["antiarrhythmic"])

    json_str = report.model_dump_json()
    parsed = json.loads(json_str)
    rebuilt = type(report).model_validate_json(json_str)
    assert rebuilt == report
    assert parsed["method"] == "baseline"

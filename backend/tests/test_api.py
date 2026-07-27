"""Phase 16: FastAPI route tests using TestClient + fake data
(build_fake_app_state), overriding the AppState dependency per FastAPI's
standard testing pattern -- no real MIMIC data or live OpenAI calls."""
import json

import pytest
from fastapi.testclient import TestClient

from cds.agent import LLMTurn, MockLLMProvider, ToolCallRequest, MAX_TOOL_CALLS
from cds_api.app_state import build_fake_app_state
from cds_api.dependencies import get_app_state
from cds_api.routes import app


@pytest.fixture
def client():
    state = build_fake_app_state()
    app.dependency_overrides[get_app_state] = lambda: state
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def agent_client():
    """A client whose agent calls always succeed via a scripted MockLLMProvider."""

    def factory():
        return MockLLMProvider(
            [
                LLMTurn(
                    content=json.dumps(
                        {
                            "recommendations": [
                                {"medication_class": "antiarrhythmic", "action": "start", "rationale": "Rate control.", "confidence": 0.7}
                            ]
                        }
                    ),
                    tool_calls=[],
                )
            ]
        )

    state = build_fake_app_state(llm_provider_factory=factory)
    app.dependency_overrides[get_app_state] = lambda: state
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_cohort_returns_only_test_split(client):
    response = client.get("/cohort")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5  # build_fake_app_state puts subjects 21-25 in test
    assert all(21 <= p["subject_id"] <= 25 for p in body)


def test_get_timeline_found(client):
    response = client.get("/patients/21/admissions/1021/timeline")
    assert response.status_code == 200
    assert response.json()["subject_id"] == 21


def test_get_timeline_not_found(client):
    response = client.get("/patients/999999/admissions/999999/timeline")
    assert response.status_code == 404


def test_get_clinical_state_found(client):
    response = client.get("/patients/21/admissions/1021/clinical-state")
    assert response.status_code == 200
    assert response.json()["hadm_id"] == 1021


def test_get_clinical_state_not_found(client):
    response = client.get("/patients/999999/admissions/999999/clinical-state")
    assert response.status_code == 404


def test_predict_baseline(client):
    response = client.post("/predict/baseline", json={"subject_id": 21, "hadm_id": 1021})
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "baseline"
    assert body["subject_id"] == 21


def test_predict_model(client):
    response = client.post("/predict/model", json={"subject_id": 21, "hadm_id": 1021})
    assert response.status_code == 200
    assert response.json()["method"] == "trained_model"


def test_predict_baseline_not_found(client):
    response = client.post("/predict/baseline", json={"subject_id": 999999, "hadm_id": 999999})
    assert response.status_code == 404


def test_predict_agent(agent_client):
    response = agent_client.post("/predict/agent", json={"subject_id": 21, "hadm_id": 1021})
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "agent"
    assert body["recommended_medications"][0]["medication_class"] == "antiarrhythmic"


def test_predict_agent_exceeding_tool_call_budget_returns_502():
    """A model that never converges (e.g. one tool call per turn, indefinitely)
    surfaces as a clean 502 with a readable detail -- not an unhandled 500."""

    def factory():
        never_ending_turn = LLMTurn(
            content=None,
            tool_calls=[ToolCallRequest(id="call-0", name="search_similar_patient_profiles", arguments={"top_k": 5})],
        )
        return MockLLMProvider([never_ending_turn] * (MAX_TOOL_CALLS + 1))

    state = build_fake_app_state(llm_provider_factory=factory)
    app.dependency_overrides[get_app_state] = lambda: state
    try:
        client = TestClient(app)
        response = client.post("/predict/agent", json={"subject_id": 21, "hadm_id": 1021})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "max_tool_calls" in response.json()["detail"]


def test_synthetic_new_patient_returns_usable_new_patient_request(client):
    response = client.get("/synthetic/new-patient")
    assert response.status_code == 200
    body = response.json()
    assert 18 <= body["age"] <= 95
    assert body["admission_reason"] != "Hypertension"


def test_synthetic_new_patient_payload_passes_new_patient_validation(agent_client):
    synthetic = agent_client.get("/synthetic/new-patient").json()
    # the synthetic example should itself pass straight into
    # predict/agent/new-patient's validation (same vocabulary, by construction)
    submit = agent_client.post("/predict/agent/new-patient", json=synthetic)
    assert submit.status_code == 200


def test_predict_agent_new_patient_valid(agent_client):
    response = agent_client.post(
        "/predict/agent/new-patient",
        json={
            "age": 70, "gender": "F", "admission_reason": "Cardiac arrhythmias",
            "hypertension_status": "confirmed_chronic", "current_medication_classes": ["ace inhibitor"],
            "recent_creatinine": 1.0,
        },
    )
    assert response.status_code == 200
    assert response.json()["method"] == "agent"


def test_predict_agent_new_patient_unknown_medication_class_rejected(agent_client):
    response = agent_client.post(
        "/predict/agent/new-patient",
        json={
            "age": 70, "gender": "F", "admission_reason": "Cardiac arrhythmias",
            "hypertension_status": "confirmed_chronic", "current_medication_classes": ["not-a-real-class"],
        },
    )
    assert response.status_code == 400


def test_evaluate_without_agent(client):
    response = client.post("/evaluate", json={"include_agent": False})
    assert response.status_code == 200
    body = response.json()
    methods = {row["method"] for row in body["comparison_table"]}
    assert methods == {"baseline", "trained_model", "agent"}


def test_evaluate_with_hadm_ids_filter(client):
    response = client.post("/evaluate", json={"hadm_ids": [1021], "include_agent": False})
    assert response.status_code == 200
    body = response.json()
    baseline_eval = next(ev for ev in body["evaluations"] if ev["method"] == "baseline")
    assert baseline_eval["n_admissions_evaluated"] == 1


def test_evaluate_no_matching_admissions_returns_400(client):
    response = client.post("/evaluate", json={"hadm_ids": [999999]})
    assert response.status_code == 400


def test_explain_baseline(client):
    response = client.get("/patients/21/admissions/1021", params={"method": "baseline"})
    # (not a real route -- sanity check that unrelated paths still 404 cleanly)
    assert response.status_code == 404

    response = client.get("/explain/21/1021", params={"method": "baseline"})
    assert response.status_code == 200
    assert response.json()["method"] == "baseline"


def test_explain_trained_model(client):
    response = client.get("/explain/21/1021", params={"method": "trained_model"})
    assert response.status_code == 200
    assert len(response.json()["feature_importances"]) > 0


def test_explain_invalid_method(client):
    response = client.get("/explain/21/1021", params={"method": "not-a-method"})
    assert response.status_code == 400


def test_explain_not_found(client):
    response = client.get("/explain/999999/999999", params={"method": "baseline"})
    assert response.status_code == 404

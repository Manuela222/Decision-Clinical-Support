"""Phase 9: trained model tests (feature extraction, training, prediction,
persistence). Uses a small fully-synthetic training set -- these tests check
interface correctness and shape/range validity, not model accuracy (which
would be flaky on this little data by construction)."""
import random
from datetime import datetime, timezone

import pytest

from cds.model import (
    ModelCard,
    build_structured_features,
    load_model_artifact,
    load_model_card,
    recommend_trained_model,
    save_model_artifact,
    train_model,
)
from cds.schemas import ClinicalState, Gender, HypertensionStatus, LabValueSummary, RecommendationMethod


def _clinical_state(idx: int, admission_reason: str, creatinine: float) -> ClinicalState:
    return ClinicalState(
        subject_id=idx,
        hadm_id=1000 + idx,
        age=40 + (idx % 40),
        gender=Gender.F if idx % 2 == 0 else Gender.M,
        admission_reason=admission_reason,
        hypertension_status=HypertensionStatus.CONFIRMED_CHRONIC,
        hypertension_evidence=[],
        recent_blood_pressure_labs=[],
        current_antihypertensive_medications=[],
        active_conditions=[admission_reason, "Hypertension"],
        chronic_conditions=[admission_reason, "Hypertension"],
        recent_labs=[
            LabValueSummary(
                itemid=1, label="Creatinine", value=creatinine, unit="mg/dL", flag=None,
                charttime=None, evidence_id=f"ev-{idx}",
            )
        ],
        current_medications=[],
        prior_medications=[],
        relevant_note_sections=[],
        missing_information=[],
        generated_at=datetime.now(timezone.utc),
    )


def _training_set(n=30, seed=0):
    rng = random.Random(seed)
    states, labels = [], []
    reasons_and_labels = [
        ("Cardiac arrhythmias", ["antiarrhythmic"]),
        ("Diabetes, uncomplicated", ["antidiabetic - biguanide"]),
    ]
    for i in range(n):
        reason, base_labels = reasons_and_labels[i % 2]
        creatinine = rng.uniform(0.6, 1.3)
        states.append(_clinical_state(i, reason, creatinine))
        labels.append(list(base_labels))
    return states, labels


# --- build_structured_features -------------------------------------------

def test_build_structured_features_extracts_lab_value():
    states = [_clinical_state(1, "Cardiac arrhythmias", 1.4)]
    df = build_structured_features(states)
    assert df.loc[0, "lab_creatinine"] == pytest.approx(1.4)
    assert df.loc[0, "admission_reason"] == "Cardiac arrhythmias"


def test_build_structured_features_missing_lab_is_nan():
    state = _clinical_state(1, "Cardiac arrhythmias", 1.0)
    state.recent_labs = []
    df = build_structured_features([state])
    assert df.loc[0, "lab_creatinine"] != df.loc[0, "lab_creatinine"]  # NaN != NaN


# --- train_model -----------------------------------------------------------

def test_train_model_produces_valid_artifact():
    states, labels = _training_set(n=30)
    artifact = train_model(states, labels, seed=42)
    assert isinstance(artifact.model_card, ModelCard)
    assert artifact.model_card.n_training_admissions == 30
    assert artifact.model_card.n_validation_admissions == 6  # round(30*0.2)
    assert set(artifact.model_card.label_classes) == {"antiarrhythmic", "antidiabetic - biguanide"}
    assert 0.0 <= artifact.model_card.validation_metrics["micro_f1"] <= 1.0


def test_train_model_mismatched_lengths_raises():
    states, labels = _training_set(n=10)
    with pytest.raises(ValueError, match="same length"):
        train_model(states, labels[:-1])


def test_train_model_too_few_admissions_raises():
    states, labels = _training_set(n=3)
    with pytest.raises(ValueError, match="at least 5"):
        train_model(states, labels)


def test_train_model_known_limitations_documented():
    states, labels = _training_set(n=10)
    artifact = train_model(states, labels, seed=1)
    limitations_text = " ".join(artifact.model_card.known_limitations).lower()
    assert "blood pressure" in limitations_text
    assert "tf-idf" in limitations_text


# --- recommend_trained_model -------------------------------------------------

@pytest.fixture
def artifact():
    states, labels = _training_set(n=30)
    return train_model(states, labels, seed=42)


def test_recommend_trained_model_returns_valid_result(artifact):
    query = _clinical_state(999, "Cardiac arrhythmias", 1.0)
    result = recommend_trained_model(query, artifact, threshold=0.0)
    assert result.method == RecommendationMethod.TRAINED_MODEL
    assert result.subject_id == query.subject_id
    assert result.hadm_id == query.hadm_id
    for rec in result.recommended_medications:
        assert 0.0 <= rec.confidence <= 1.0
        assert rec.hypertension_reasoning
        assert rec.evidence_ids == []


def test_recommend_trained_model_high_threshold_yields_no_recommendations(artifact):
    query = _clinical_state(999, "Cardiac arrhythmias", 1.0)
    result = recommend_trained_model(query, artifact, threshold=1.01)
    assert result.recommended_medications == []


def test_recommend_trained_model_respects_top_k(artifact):
    query = _clinical_state(999, "Cardiac arrhythmias", 1.0)
    result = recommend_trained_model(query, artifact, threshold=0.0, top_k=1)
    assert len(result.recommended_medications) <= 1


# --- persistence -------------------------------------------------------------

def test_save_and_load_model_artifact_round_trips(tmp_path, artifact):
    path = tmp_path / "model.joblib"
    save_model_artifact(artifact, path)

    loaded = load_model_artifact(path)
    query = _clinical_state(999, "Diabetes, uncomplicated", 1.0)
    result_before = recommend_trained_model(query, artifact, threshold=0.0)
    result_after = recommend_trained_model(query, loaded, threshold=0.0)
    assert [r.medication_class for r in result_before.recommended_medications] == [
        r.medication_class for r in result_after.recommended_medications
    ]

    card = load_model_card(path)
    assert card.model_version == artifact.model_card.model_version
    assert card.n_training_admissions == artifact.model_card.n_training_admissions


def test_load_model_artifact_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained model artifact"):
        load_model_artifact(tmp_path / "nope.joblib")


def test_load_model_card_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model card found"):
        load_model_card(tmp_path / "nope.joblib")

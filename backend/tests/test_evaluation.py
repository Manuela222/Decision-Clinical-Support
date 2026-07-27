"""Phase 14: evaluation tests."""
from datetime import datetime, timezone

import pytest

from cds.evaluation import (
    build_comparison_table,
    evaluate_all_methods,
    evaluate_medication_recommendations,
    predicted_classes_from_result,
)
from cds.schemas import MedicationAction, MedicationRecommendation, RecommendationMethod, RecommendationResult


def _rec(medication_class: str) -> MedicationRecommendation:
    return MedicationRecommendation(
        medication_class=medication_class,
        action=MedicationAction.START,
        rationale="test",
        evidence_ids=[],
        confidence=0.5,
        hypertension_compatible=True,
        hypertension_reasoning="test",
    )


def _result(method: RecommendationMethod, subject_id: int, hadm_id: int, classes) -> RecommendationResult:
    return RecommendationResult(
        subject_id=subject_id,
        hadm_id=hadm_id,
        method=method,
        recommended_medications=[_rec(c) for c in classes],
        safety_warnings=[],
        reasoning_trace=[],
        model_version="test",
        generated_at=datetime.now(timezone.utc),
    )


# --- evaluate_medication_recommendations ------------------------------------

def test_evaluate_medication_recommendations_basic():
    predicted = [["a", "b"], ["c"]]
    ground_truth = [["a"], ["c", "d"]]
    result = evaluate_medication_recommendations(predicted, ground_truth, RecommendationMethod.BASELINE)

    assert result.n_admissions_evaluated == 2
    assert result.per_admission[0].matched_classes == ["a"]
    assert result.per_admission[0].extra_classes == ["b"]
    assert result.per_admission[0].missed_classes == []
    assert result.per_admission[1].matched_classes == ["c"]
    assert result.per_admission[1].missed_classes == ["d"]

    # micro: tp=2 (a, c), fp=1 (b), fn=1 (d)
    assert result.micro_precision == pytest.approx(2 / 3)
    assert result.micro_recall == pytest.approx(2 / 3)


def test_evaluate_medication_recommendations_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        evaluate_medication_recommendations([["a"]], [["a"], ["b"]], RecommendationMethod.BASELINE)


def test_evaluate_medication_recommendations_carries_subject_hadm_ids():
    result = evaluate_medication_recommendations(
        [["a"]], [["a"]], RecommendationMethod.AGENT, subject_ids=[42], hadm_ids=[100]
    )
    assert result.per_admission[0].subject_id == 42
    assert result.per_admission[0].hadm_id == 100


# --- predicted_classes_from_result ------------------------------------------

def test_predicted_classes_from_result():
    result = _result(RecommendationMethod.BASELINE, 1, 100, ["statin", "antiarrhythmic"])
    assert predicted_classes_from_result(result) == ["statin", "antiarrhythmic"]


# --- build_comparison_table --------------------------------------------------

def test_build_comparison_table_shape():
    ev_a = evaluate_medication_recommendations([["a"]], [["a"]], RecommendationMethod.BASELINE)
    ev_b = evaluate_medication_recommendations([["a"]], [["b"]], RecommendationMethod.AGENT)
    table = build_comparison_table([ev_a, ev_b])
    assert list(table["method"]) == ["baseline", "agent"]
    assert table.loc[0, "micro_f1"] == pytest.approx(1.0)
    assert table.loc[1, "micro_f1"] == pytest.approx(0.0)


# --- evaluate_all_methods ------------------------------------------------------

def test_evaluate_all_methods_produces_comparison_report():
    ground_truth = [["antiarrhythmic"], ["antidiabetic - biguanide", "statin"]]
    baseline_results = [
        _result(RecommendationMethod.BASELINE, 1, 100, ["antiarrhythmic"]),
        _result(RecommendationMethod.BASELINE, 2, 101, ["antidiabetic - biguanide"]),
    ]
    model_results = [
        _result(RecommendationMethod.TRAINED_MODEL, 1, 100, ["antiarrhythmic"]),
        _result(RecommendationMethod.TRAINED_MODEL, 2, 101, ["antidiabetic - biguanide", "statin"]),
    ]
    agent_results = [
        _result(RecommendationMethod.AGENT, 1, 100, []),
        _result(RecommendationMethod.AGENT, 2, 101, ["antidiabetic - biguanide", "statin"]),
    ]

    report = evaluate_all_methods(ground_truth, baseline_results, model_results, agent_results)

    assert [ev.method for ev in report.evaluations] == [
        RecommendationMethod.BASELINE,
        RecommendationMethod.TRAINED_MODEL,
        RecommendationMethod.AGENT,
    ]
    assert list(report.table["method"]) == ["baseline", "trained_model", "agent"]
    # model got everything right -> perfect scores
    model_row = report.table[report.table["method"] == "trained_model"].iloc[0]
    assert model_row["micro_f1"] == pytest.approx(1.0)
    # agent missed admission 1 entirely -> imperfect recall
    agent_row = report.table[report.table["method"] == "agent"].iloc[0]
    assert agent_row["micro_recall"] < 1.0


def test_evaluate_all_methods_raises_on_length_mismatch():
    ground_truth = [["a"], ["b"]]
    short = [_result(RecommendationMethod.BASELINE, 1, 100, ["a"])]
    full = [
        _result(RecommendationMethod.TRAINED_MODEL, 1, 100, ["a"]),
        _result(RecommendationMethod.TRAINED_MODEL, 2, 101, ["b"]),
    ]
    with pytest.raises(ValueError, match="baseline_results"):
        evaluate_all_methods(ground_truth, short, full, full)


def test_evaluate_all_methods_raises_on_misaligned_admissions():
    ground_truth = [["a"], ["b"]]
    baseline_results = [
        _result(RecommendationMethod.BASELINE, 1, 100, ["a"]),
        _result(RecommendationMethod.BASELINE, 2, 101, ["b"]),
    ]
    # model_results admission order swapped relative to baseline/agent
    model_results = [
        _result(RecommendationMethod.TRAINED_MODEL, 2, 101, ["b"]),
        _result(RecommendationMethod.TRAINED_MODEL, 1, 100, ["a"]),
    ]
    agent_results = [
        _result(RecommendationMethod.AGENT, 1, 100, ["a"]),
        _result(RecommendationMethod.AGENT, 2, 101, ["b"]),
    ]
    with pytest.raises(ValueError, match="misaligned"):
        evaluate_all_methods(ground_truth, baseline_results, model_results, agent_results)

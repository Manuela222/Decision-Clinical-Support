"""Phase 14: evaluate_medication_recommendations wraps Phase 9's shared
`compute_multilabel_metrics` into the Phase 1 EvaluationResult/
PerAdmissionEvaluation schema, plus a runner that scores baseline, trained
model, and agent against the identical test-set admissions with the
identical metric function and produces one comparison table."""
from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from ..schemas import EvaluationResult, PerAdmissionEvaluation, RecommendationMethod, RecommendationResult
from .metrics import compute_multilabel_metrics


def predicted_classes_from_result(result: RecommendationResult) -> List[str]:
    return [rec.medication_class for rec in result.recommended_medications]


def evaluate_medication_recommendations(
    predicted: Sequence[Sequence[str]],
    ground_truth_meds: Sequence[Sequence[str]],
    method: RecommendationMethod,
    subject_ids: Sequence[int] = (),
    hadm_ids: Sequence[int] = (),
) -> EvaluationResult:
    """Score `predicted` medication-class sets against `ground_truth_meds`
    (same length, same admission order) for one `method`, using the exact
    same metric function every method is scored with."""
    if len(predicted) != len(ground_truth_meds):
        raise ValueError(f"predicted ({len(predicted)}) and ground_truth_meds ({len(ground_truth_meds)}) must be the same length.")

    metrics = compute_multilabel_metrics(ground_truth_meds, predicted)

    per_admission = [
        PerAdmissionEvaluation(
            subject_id=subject_ids[i] if i < len(subject_ids) else -1,
            hadm_id=hadm_ids[i] if i < len(hadm_ids) else -1,
            method=method,
            predicted_classes=admission["predicted_classes"],
            ground_truth_classes=admission["ground_truth_classes"],
            matched_classes=admission["matched_classes"],
            missed_classes=admission["missed_classes"],
            extra_classes=admission["extra_classes"],
            precision=admission["precision"],
            recall=admission["recall"],
            f1=admission["f1"],
        )
        for i, admission in enumerate(metrics["per_admission"])
    ]

    return EvaluationResult(
        method=method,
        n_admissions_evaluated=metrics["n_admissions_evaluated"],
        micro_precision=metrics["micro_precision"],
        micro_recall=metrics["micro_recall"],
        micro_f1=metrics["micro_f1"],
        macro_precision=metrics["macro_precision"],
        macro_recall=metrics["macro_recall"],
        macro_f1=metrics["macro_f1"],
        per_admission=per_admission,
    )


def build_comparison_table(evaluations: List[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": ev.method.value,
                "n_admissions_evaluated": ev.n_admissions_evaluated,
                "micro_precision": ev.micro_precision,
                "micro_recall": ev.micro_recall,
                "micro_f1": ev.micro_f1,
                "macro_precision": ev.macro_precision,
                "macro_recall": ev.macro_recall,
                "macro_f1": ev.macro_f1,
            }
            for ev in evaluations
        ]
    )


@dataclass
class ComparisonReport:
    evaluations: List[EvaluationResult]
    table: pd.DataFrame


def evaluate_all_methods(
    ground_truth_meds: List[List[str]],
    baseline_results: List[RecommendationResult],
    model_results: List[RecommendationResult],
    agent_results: List[RecommendationResult],
) -> ComparisonReport:
    """Run baseline/trained-model/agent RecommendationResults (already
    computed, over the SAME test-set admissions in the SAME order) through
    the identical metric function and produce one comparison table.

    Validates that all three result lists (and ground_truth_meds) actually
    line up admission-for-admission before scoring — a silent misalignment
    here would make the whole comparison meaningless.
    """
    n = len(ground_truth_meds)
    for name, results in [("baseline_results", baseline_results), ("model_results", model_results), ("agent_results", agent_results)]:
        if len(results) != n:
            raise ValueError(f"{name} has {len(results)} entries, expected {n} (len(ground_truth_meds)).")

    for i in range(n):
        hadm_ids_at_i = {baseline_results[i].hadm_id, model_results[i].hadm_id, agent_results[i].hadm_id}
        if len(hadm_ids_at_i) != 1:
            raise ValueError(
                f"Result lists are misaligned at index {i}: hadm_ids "
                f"{baseline_results[i].hadm_id}, {model_results[i].hadm_id}, {agent_results[i].hadm_id} "
                f"must all match the same admission for a valid comparison."
            )

    subject_ids = [r.subject_id for r in baseline_results]
    hadm_ids = [r.hadm_id for r in baseline_results]

    evaluations = [
        evaluate_medication_recommendations(
            [predicted_classes_from_result(r) for r in results], ground_truth_meds, method, subject_ids, hadm_ids
        )
        for method, results in [
            (RecommendationMethod.BASELINE, baseline_results),
            (RecommendationMethod.TRAINED_MODEL, model_results),
            (RecommendationMethod.AGENT, agent_results),
        ]
    ]

    return ComparisonReport(evaluations=evaluations, table=build_comparison_table(evaluations))

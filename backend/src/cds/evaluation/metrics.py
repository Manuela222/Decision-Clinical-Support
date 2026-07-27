"""Shared multi-label precision/recall/F1 (micro + macro) metric.

Pulled forward from Phase 14 (Evaluation) out of necessity: Phase 9's
trained model needs to "report validation metrics" on its own internal
holdout, using the exact same metric definition Phase 14 will later use to
compare baseline vs. trained model vs. agent. Phase 14 wraps this function
to build `EvaluationResult`/`PerAdmissionEvaluation`; it does not
reimplement the math.
"""
from collections import Counter
from typing import Any, Dict, List, Sequence


def _safe_div(numerator: float, denominator: float, default: float) -> float:
    return numerator / denominator if denominator else default


def compute_multilabel_metrics(
    y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]
) -> Dict[str, Any]:
    """Compare predicted vs. ground-truth medication-class sets across a
    batch of admissions. Returns micro/macro precision/recall/F1 plus a
    per-admission breakdown (matched/missed/extra classes)."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must be the same length.")

    per_admission: List[Dict[str, Any]] = []
    tp_total = fp_total = fn_total = 0
    per_class_tp: Counter = Counter()
    per_class_fp: Counter = Counter()
    per_class_fn: Counter = Counter()

    for true_classes, pred_classes in zip(y_true, y_pred):
        true_set, pred_set = set(true_classes), set(pred_classes)
        matched = sorted(true_set & pred_set)
        missed = sorted(true_set - pred_set)
        extra = sorted(pred_set - true_set)
        tp, fp, fn = len(matched), len(extra), len(missed)

        precision = _safe_div(tp, tp + fp, default=1.0 if fn == 0 else 0.0)
        recall = _safe_div(tp, tp + fn, default=1.0)
        f1 = _safe_div(2 * precision * recall, precision + recall, default=0.0)

        per_admission.append(
            {
                "predicted_classes": sorted(pred_set),
                "ground_truth_classes": sorted(true_set),
                "matched_classes": matched,
                "missed_classes": missed,
                "extra_classes": extra,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        tp_total += tp
        fp_total += fp
        fn_total += fn
        for c in matched:
            per_class_tp[c] += 1
        for c in extra:
            per_class_fp[c] += 1
        for c in missed:
            per_class_fn[c] += 1

    micro_precision = _safe_div(tp_total, tp_total + fp_total, default=0.0)
    micro_recall = _safe_div(tp_total, tp_total + fn_total, default=0.0)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall, default=0.0)

    all_classes = sorted(set(per_class_tp) | set(per_class_fp) | set(per_class_fn))
    per_class_precisions, per_class_recalls, per_class_f1s = [], [], []
    for c in all_classes:
        tp, fp, fn = per_class_tp[c], per_class_fp[c], per_class_fn[c]
        p = _safe_div(tp, tp + fp, default=0.0)
        r = _safe_div(tp, tp + fn, default=0.0)
        f = _safe_div(2 * p * r, p + r, default=0.0)
        per_class_precisions.append(p)
        per_class_recalls.append(r)
        per_class_f1s.append(f)

    macro_precision = sum(per_class_precisions) / len(per_class_precisions) if per_class_precisions else 0.0
    macro_recall = sum(per_class_recalls) / len(per_class_recalls) if per_class_recalls else 0.0
    macro_f1 = sum(per_class_f1s) / len(per_class_f1s) if per_class_f1s else 0.0

    return {
        "n_admissions_evaluated": len(y_true),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_admission": per_admission,
    }

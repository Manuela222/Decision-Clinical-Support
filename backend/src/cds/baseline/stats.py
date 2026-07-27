"""Precompute 'most common discharge medication class per admission
condition' statistics from the training split — the baseline's only
"model"."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class DiagnosisMedicationStats:
    # admission_reason -> [(medication_class, frequency in [0,1]), ...] sorted desc by frequency
    by_admission_reason: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    # cohort-wide fallback for an admission_reason unseen in training
    overall: List[Tuple[str, float]] = field(default_factory=list)


def compute_diagnosis_medication_stats(
    admission_reasons: Sequence[str],
    ground_truth_classes: Sequence[Sequence[str]],
) -> DiagnosisMedicationStats:
    """Build baseline statistics from the training split.

    `ground_truth_classes[i]` is the set of discharge medication classes for
    training admission `i`, and must already exclude antihypertensive-tagged
    classes (see `cds.medications.is_antihypertensive_class`) — this
    function trusts its input rather than re-filtering, matching the
    project's convention that antihypertensive classes are never part of
    the prediction label space.
    """
    if len(admission_reasons) != len(ground_truth_classes):
        raise ValueError(
            f"admission_reasons ({len(admission_reasons)}) and ground_truth_classes "
            f"({len(ground_truth_classes)}) must be the same length."
        )

    counts_by_reason: Dict[str, Counter] = defaultdict(Counter)
    n_by_reason: Dict[str, int] = defaultdict(int)
    overall_counts: Counter = Counter()
    n_total = len(admission_reasons)

    for reason, classes in zip(admission_reasons, ground_truth_classes):
        n_by_reason[reason] += 1
        for cls in set(classes):
            counts_by_reason[reason][cls] += 1
            overall_counts[cls] += 1

    by_admission_reason = {
        reason: sorted(
            ((cls, count / n_by_reason[reason]) for cls, count in counter.items()),
            key=lambda item: (-item[1], item[0]),
        )
        for reason, counter in counts_by_reason.items()
    }
    overall = (
        sorted(
            ((cls, count / n_total) for cls, count in overall_counts.items()),
            key=lambda item: (-item[1], item[0]),
        )
        if n_total
        else []
    )

    return DiagnosisMedicationStats(by_admission_reason=by_admission_reason, overall=overall)

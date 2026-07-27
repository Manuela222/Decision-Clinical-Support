"""Local, rule-based drug-interaction check (Phase 12's `check_drug_interactions`
MCP tool wraps this). Deliberately out of scope for Phase 10's recommendation
safety checker, which covers duplicate/hypertension/renal/indication/evidence
only — interactions are checked per-candidate against the patient's current
medication list, one MCP tool call at a time, not as a batch pass over a
whole recommendation set.

This is a small, curated table of well-established, clinically significant
interactions relevant to this project's medication-class vocabulary — not an
exhaustive drug-interaction database.
"""
from typing import List, NamedTuple

from pydantic import BaseModel

from ..schemas import ClinicalState, SafetySeverity


class InteractionRule(NamedTuple):
    class_a: str
    class_b: str
    severity: SafetySeverity
    message: str


INTERACTION_RULES: List[InteractionRule] = [
    InteractionRule(
        "nsaid", "ace inhibitor", SafetySeverity.WARNING,
        "NSAIDs can blunt the antihypertensive/renal-protective effect of ACE inhibitors and increase renal risk when combined.",
    ),
    InteractionRule(
        "nsaid", "arb", SafetySeverity.WARNING,
        "NSAIDs can blunt the antihypertensive effect of ARBs and increase renal risk when combined.",
    ),
    InteractionRule(
        "nsaid", "loop diuretic", SafetySeverity.WARNING,
        "NSAIDs can blunt diuretic effect and increase renal risk — part of the 'triple whammy' when an "
        "ACE inhibitor/ARB is also present.",
    ),
    InteractionRule(
        "ace inhibitor", "arb", SafetySeverity.CRITICAL,
        "Combining an ACE inhibitor and an ARB is not recommended: increased hyperkalemia/renal-impairment "
        "risk with no added benefit.",
    ),
    InteractionRule(
        "anticoagulant", "antiplatelet", SafetySeverity.WARNING,
        "Combining an anticoagulant and an antiplatelet increases bleeding risk.",
    ),
    InteractionRule(
        "anticoagulant", "nsaid", SafetySeverity.CRITICAL,
        "Combining an anticoagulant and an NSAID substantially increases bleeding risk.",
    ),
    InteractionRule(
        "beta blocker", "calcium channel blocker", SafetySeverity.WARNING,
        "Combining a beta blocker and a calcium channel blocker can cause additive bradycardia/AV block.",
    ),
]


class DrugInteractionWarning(BaseModel):
    candidate_medication_class: str
    interacting_medication_class: str
    severity: SafetySeverity
    message: str


def check_drug_interactions(
    candidate_medication_class: str, clinical_state: ClinicalState
) -> List[DrugInteractionWarning]:
    """Check `candidate_medication_class` against every class already in
    `clinical_state.current_medications`, using the curated rule table above."""
    current_classes = {m.medication_class for m in clinical_state.current_medications}
    warnings: List[DrugInteractionWarning] = []
    for rule in INTERACTION_RULES:
        if candidate_medication_class == rule.class_a and rule.class_b in current_classes:
            other = rule.class_b
        elif candidate_medication_class == rule.class_b and rule.class_a in current_classes:
            other = rule.class_a
        else:
            continue
        warnings.append(
            DrugInteractionWarning(
                candidate_medication_class=candidate_medication_class,
                interacting_medication_class=other,
                severity=rule.severity,
                message=rule.message,
            )
        )
    return warnings

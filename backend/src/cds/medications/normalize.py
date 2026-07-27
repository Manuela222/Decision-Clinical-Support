"""Medication name normalization: strip dosage/route/salt noise, apply
synonym mapping, and produce a stable base-ingredient name for class lookup."""
import re
from typing import List, Optional

from .dictionary import MedicationDictionary, default_dictionary

_PAREN_RE = re.compile(r"\([^)]*\)")
_DOSAGE_RE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|meq|unit|units|%)\b", re.IGNORECASE)
_SALT_SUFFIX_RE = re.compile(
    r"\b(hcl|sodium|na succ|sulfate|bisulfate|tartrate|succinate|citrate|besylate|"
    r"phosphate|acetate|maleate|mesylate|bromide|gluconate|hydrobromide|fumarate)\b",
    re.IGNORECASE,
)
_FORM_ROUTE_RE = re.compile(
    r"\b(ec|xl|sr|er|cr|iv|po|liquid|tablet|capsule|extended release|neb soln|neb|soln|"
    r"inj|injection|oral|mdi|patch|ophth)\b",
    re.IGNORECASE,
)
_NON_WORD_RE = re.compile(r"[^a-z0-9\s\-]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_medication_name(raw_name: str, dictionary: Optional[MedicationDictionary] = None) -> str:
    """Reduce a raw PRESCRIPTIONS.DRUG string to a stable, lookup-ready name.

    Order matters: parenthetical/dosage/salt/form-route noise is stripped
    *before* the synonym map is applied, so both brand names ("Lasix") and
    already-generic salted names ("Furosemide Sodium") normalize to the same
    base ingredient ("furosemide").
    """
    if not raw_name or not isinstance(raw_name, str):
        return ""
    dictionary = dictionary or default_dictionary()

    name = raw_name.strip().lower()
    name = _PAREN_RE.sub("", name)
    name = _DOSAGE_RE.sub("", name)
    name = _SALT_SUFFIX_RE.sub("", name)
    name = _FORM_ROUTE_RE.sub("", name)
    name = _NON_WORD_RE.sub("", name)
    name = _WHITESPACE_RE.sub(" ", name).strip()

    return dictionary.synonyms.get(name, name)


def normalize_medication_list(
    raw_names: List[str], dictionary: Optional[MedicationDictionary] = None
) -> List[str]:
    """Normalize a list of raw drug names, de-duplicating while preserving
    first-seen order (repeated orders of the same drug collapse to one)."""
    dictionary = dictionary or default_dictionary()
    normalized: List[str] = []
    for raw in raw_names:
        name = normalize_medication_name(raw, dictionary)
        if name and name not in normalized:
            normalized.append(name)
    return normalized

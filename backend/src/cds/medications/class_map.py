"""Map a normalized medication name to the fixed medication-class vocabulary
(Phase 4), with antihypertensive classes tagged separately so they can be
excluded from the prediction label space but still used for safety checks."""
from typing import Optional

from .dictionary import MedicationDictionary, default_dictionary

EXCLUDED_CLASS = "__EXCLUDE__"
UNMAPPED_CLASS = "Other/Unmapped"


def map_to_medication_class(
    normalized_name: str, dictionary: Optional[MedicationDictionary] = None
) -> Optional[str]:
    """Return the medication class for a normalized name, `EXCLUDED_CLASS`
    for non-medication entries (IV fluids, electrolytes, supplies), or
    `UNMAPPED_CLASS` if no rule in the dictionary matches. Returns None only
    for an empty/falsy input."""
    if not normalized_name:
        return None
    dictionary = dictionary or default_dictionary()

    if normalized_name in dictionary.exclude_entirely:
        return EXCLUDED_CLASS
    if normalized_name in dictionary.classes:
        return dictionary.classes[normalized_name]

    for key, cls in dictionary.classes.items():
        if key in normalized_name:
            return cls
    for key in dictionary.exclude_entirely:
        if key in normalized_name:
            return EXCLUDED_CLASS

    return UNMAPPED_CLASS


def is_antihypertensive_class(medication_class: str, dictionary: Optional[MedicationDictionary] = None) -> bool:
    dictionary = dictionary or default_dictionary()
    return medication_class in dictionary.antihypertensive_classes

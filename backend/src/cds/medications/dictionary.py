"""Loading of the medication synonym + class dictionary (YAML or JSON)."""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Set

import yaml

DEFAULT_DICTIONARY_PATH = Path(__file__).parent / "data" / "default_dictionary.yaml"


class MedicationDictionaryError(Exception):
    """Raised when a medication dictionary file is missing or malformed."""


@dataclass(frozen=True)
class MedicationDictionary:
    synonyms: Dict[str, str] = field(default_factory=dict)
    classes: Dict[str, str] = field(default_factory=dict)
    antihypertensive_classes: Set[str] = field(default_factory=set)
    exclude_entirely: Set[str] = field(default_factory=set)


def load_medication_dictionary(path: "Path | str") -> MedicationDictionary:
    """Load a synonym/class dictionary from a YAML or JSON file.

    Expected top-level keys (all optional): `synonyms` (map), `classes`
    (map), `antihypertensive_classes` (list), `exclude_entirely` (list).
    """
    path = Path(path)
    if not path.is_file():
        raise MedicationDictionaryError(f"Medication dictionary file not found: '{path}'.")

    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise MedicationDictionaryError(
            f"Unsupported medication dictionary file extension '{path.suffix}' for '{path}'. "
            "Use .yaml, .yml, or .json."
        )

    if not isinstance(data, dict):
        raise MedicationDictionaryError(f"Medication dictionary file '{path}' must contain a top-level mapping.")

    return MedicationDictionary(
        synonyms={str(k).lower(): str(v).lower() for k, v in (data.get("synonyms") or {}).items()},
        classes={str(k).lower(): str(v).lower() for k, v in (data.get("classes") or {}).items()},
        antihypertensive_classes={str(c).lower() for c in (data.get("antihypertensive_classes") or [])},
        exclude_entirely={str(c).lower() for c in (data.get("exclude_entirely") or [])},
    )


@lru_cache(maxsize=1)
def default_dictionary() -> MedicationDictionary:
    return load_medication_dictionary(DEFAULT_DICTIONARY_PATH)

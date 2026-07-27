"""Phase 4: medication normalization + class mapping tests."""
import pytest

from cds.medications import (
    EXCLUDED_CLASS,
    MedicationDictionary,
    MedicationDictionaryError,
    UNMAPPED_CLASS,
    is_antihypertensive_class,
    load_medication_dictionary,
    map_to_medication_class,
    normalize_medication_list,
    normalize_medication_name,
)


# --- normalize_medication_name -----------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("lasix", "furosemide"),
        ("Lasix 40mg PO", "furosemide"),
        ("Tylenol", "acetaminophen"),
        ("Coumadin", "warfarin"),
        ("Zocor", "simvastatin"),
        ("Lopressor", "metoprolol"),
        ("Glucophage", "metformin"),
    ],
)
def test_synonym_mapping(raw, expected):
    assert normalize_medication_name(raw) == expected


def test_strips_dosage_and_route_noise():
    assert normalize_medication_name("Metoprolol Tartrate 25mg Tablet") == "metoprolol"
    assert normalize_medication_name("Furosemide 40mg PO") == "furosemide"


def test_strips_salt_suffix_without_losing_synonym_lookup():
    # Salt suffixes must be stripped *before* class lookup so "Fentanyl
    # Citrate" and "Atropine Sulfate" resolve to their base ingredient.
    assert normalize_medication_name("Fentanyl Citrate") == "fentanyl"
    assert normalize_medication_name("Atropine Sulfate") == "atropine"


def test_strips_parenthetical_noise():
    assert normalize_medication_name("D5W (EXCEL BAG)") == "dextrose"


def test_empty_and_non_string_input():
    assert normalize_medication_name("") == ""
    assert normalize_medication_name(None) == ""


# --- normalize_medication_list ------------------------------------------

def test_normalize_medication_list_dedupes_preserving_order():
    result = normalize_medication_list(["Lasix", "Tylenol", "lasix 40mg", "Coumadin"])
    assert result == ["furosemide", "acetaminophen", "warfarin"]


def test_normalize_medication_list_drops_empty_entries():
    result = normalize_medication_list(["Tylenol", "", None])
    assert result == ["acetaminophen"]


# --- map_to_medication_class / is_antihypertensive_class ----------------

def test_map_to_medication_class_known_drug():
    assert map_to_medication_class("furosemide") == "loop diuretic"
    assert map_to_medication_class("acetaminophen") == "analgesic - non-opioid"


def test_antihypertensive_classes_are_tagged():
    assert is_antihypertensive_class(map_to_medication_class("metoprolol")) is True
    assert is_antihypertensive_class(map_to_medication_class("lisinopril")) is True
    assert is_antihypertensive_class(map_to_medication_class("acetaminophen")) is False


def test_excluded_entirely_class():
    assert map_to_medication_class("dextrose") == EXCLUDED_CLASS
    assert map_to_medication_class(normalize_medication_name("Potassium Chloride")) == EXCLUDED_CLASS


def test_unknown_medication_falls_back_to_unmapped():
    assert map_to_medication_class("totally-invented-drug-xyz") == UNMAPPED_CLASS


def test_empty_normalized_name_returns_none():
    assert map_to_medication_class("") is None


# --- load_medication_dictionary (external YAML/JSON) --------------------

def test_load_custom_yaml_dictionary(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        """
        synonyms:
          brandx: fauxinolol
        classes:
          fauxinolol: fake test class
        antihypertensive_classes:
          - fake test class
        exclude_entirely:
          - saline
        """
    )
    dictionary = load_medication_dictionary(custom)
    assert isinstance(dictionary, MedicationDictionary)

    normalized = normalize_medication_name("BrandX", dictionary)
    assert normalized == "fauxinolol"
    cls = map_to_medication_class(normalized, dictionary)
    assert cls == "fake test class"
    assert is_antihypertensive_class(cls, dictionary) is True
    assert map_to_medication_class("saline", dictionary) == EXCLUDED_CLASS


def test_load_custom_json_dictionary(tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text(
        '{"synonyms": {"brandy": "fauxinolol"}, "classes": {"fauxinolol": "fake test class"}}'
    )
    dictionary = load_medication_dictionary(custom)
    assert normalize_medication_name("Brandy", dictionary) == "fauxinolol"
    assert map_to_medication_class("fauxinolol", dictionary) == "fake test class"


def test_load_dictionary_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(MedicationDictionaryError, match="not found"):
        load_medication_dictionary(tmp_path / "nope.yaml")


def test_load_dictionary_unsupported_extension_raises_clear_error(tmp_path):
    bad = tmp_path / "custom.txt"
    bad.write_text("synonyms: {}")
    with pytest.raises(MedicationDictionaryError, match="Unsupported"):
        load_medication_dictionary(bad)


def test_load_dictionary_non_mapping_content_raises_clear_error(tmp_path):
    bad = tmp_path / "custom.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(MedicationDictionaryError, match="top-level mapping"):
        load_medication_dictionary(bad)

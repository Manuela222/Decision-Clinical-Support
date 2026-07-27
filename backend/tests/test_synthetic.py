"""Phase 7: synthetic patient generator tests."""
from cds.schemas import ClinicalState, HypertensionStatus, MissingInformationFlag
from cds.synthetic import generate_synthetic_clinical_state, generate_synthetic_cohort


def test_generates_valid_clinical_state():
    state = generate_synthetic_clinical_state(seed=1)
    assert isinstance(state, ClinicalState)
    assert 18 <= state.age <= 95


def test_is_deterministic_given_same_seed():
    # generated_at is a wall-clock timestamp and legitimately varies between
    # calls; everything else should be identical for the same seed.
    a = generate_synthetic_clinical_state(seed=123)
    b = generate_synthetic_clinical_state(seed=123)
    assert a.model_dump(exclude={"generated_at"}) == b.model_dump(exclude={"generated_at"})


def test_different_seeds_produce_different_patients():
    a = generate_synthetic_clinical_state(seed=1)
    b = generate_synthetic_clinical_state(seed=2)
    assert a.model_dump(exclude={"generated_at"}) != b.model_dump(exclude={"generated_at"})


def test_subject_and_hadm_ids_are_negative_sentinels():
    state = generate_synthetic_clinical_state(seed=5)
    assert state.subject_id < 0
    assert state.hadm_id < 0


def test_admission_reason_is_never_hypertension():
    for seed in range(20):
        state = generate_synthetic_clinical_state(seed=seed)
        assert state.admission_reason != "Hypertension"


def test_confirmed_chronic_hypertension_includes_hypertension_condition():
    state = generate_synthetic_clinical_state(seed=7, hypertension_status=HypertensionStatus.CONFIRMED_CHRONIC)
    assert "Hypertension" in state.active_conditions
    assert any(m.is_antihypertensive for m in state.current_medications)


def test_not_present_hypertension_has_no_antihypertensive_bias():
    # Not present -> generator shouldn't force-add antihypertensive meds
    state = generate_synthetic_clinical_state(seed=7, hypertension_status=HypertensionStatus.NOT_PRESENT)
    assert "Hypertension" not in state.active_conditions
    assert state.current_antihypertensive_medications == []


def test_no_blood_pressure_values_fabricated():
    state = generate_synthetic_clinical_state(seed=9)
    assert state.recent_blood_pressure_labs == []
    assert MissingInformationFlag.HYPERTENSION_LABS_MISSING in state.missing_information


def test_recent_labs_are_populated_and_clearly_synthetic():
    state = generate_synthetic_clinical_state(seed=9)
    assert len(state.recent_labs) > 0
    for lab in state.recent_labs:
        assert lab.itemid < 0  # sentinel, never a real MIMIC itemid
        assert "synthetic" in lab.label.lower()


def test_missing_information_reflects_no_real_notes_or_prior_admissions():
    state = generate_synthetic_clinical_state(seed=9)
    assert MissingInformationFlag.DISCHARGE_SUMMARY_MISSING in state.missing_information
    assert MissingInformationFlag.NOTES_MISSING in state.missing_information
    assert MissingInformationFlag.PRIOR_ADMISSIONS_MISSING in state.missing_information
    assert state.relevant_note_sections == []


def test_generate_synthetic_cohort_produces_distinct_patients():
    cohort = generate_synthetic_cohort(10, seed=42)
    assert len(cohort) == 10
    subject_ids = {s.subject_id for s in cohort}
    assert len(subject_ids) == 10
    assert all(sid < 0 for sid in subject_ids)


def test_generate_synthetic_cohort_is_deterministic():
    cohort_a = generate_synthetic_cohort(5, seed=42)
    cohort_b = generate_synthetic_cohort(5, seed=42)
    dump_a = [s.model_dump(exclude={"generated_at"}) for s in cohort_a]
    dump_b = [s.model_dump(exclude={"generated_at"}) for s in cohort_b]
    assert dump_a == dump_b

"""
Phase 0, audit task 4 (exploratory only — NOT the Phase 4 deliverable).

A deliberately minimal normalization + class-mapping table covering the
highest-volume raw DRUG strings in the cohort, built by inspecting actual
value_counts() of PRESCRIPTIONS.DRUG in the filtered cohort (see
inspect step2 output). Anything not matched falls into "Other/Unmapped" so
the feasibility report can quantify how much of the label space would need
further normalization work in Phase 4.

ASSUMPTIONS FLAGGED FOR THE FEASIBILITY REPORT:
1. "Discharge medication" is not an explicit field in PRESCRIPTIONS. We
   define it heuristically as: a prescription row whose ENDDATE is null
   (still active/ongoing) OR whose ENDDATE falls on/after the admission's
   DISCHTIME date, AND whose STARTDATE is on/before DISCHTIME. This is a
   common heuristic in MIMIC-based discharge-medication studies but is an
   approximation, not ground truth (MIMIC-III PRESCRIPTIONS is sourced from
   inpatient order entry, not a discharge-reconciled medication list).
2. Antihypertensive-tagged classes (ACE inhibitors, ARBs, beta-blockers,
   calcium channel blockers, thiazide/loop diuretics, central alpha
   agonists, direct vasodilators) are excluded from the label space
   regardless of the clinical reason they were ordered (e.g. metoprolol for
   rate control vs. blood pressure), per the project's instruction to treat
   these classes as HTN-related and safety-relevant rather than as
   prediction targets. This will over-exclude in some individual cases
   (e.g. furosemide for volume overload rather than BP) but is the
   simplification the project spec calls for.
3. IV fluids/electrolyte replacement (D5W, NS, LR, Potassium Chloride,
   Magnesium Sulfate, Sodium Chloride Flush, etc.) and non-therapeutic
   supply items (Syringe, Vial) are excluded entirely (not a medication
   class relevant to discharge planning).
"""
import re
from typing import Optional

# --- normalization -----------------------------------------------------
_SYNONYMS = {
    "lasix": "furosemide",
    "tylenol": "acetaminophen",
    "coumadin": "warfarin",
    "zocor": "simvastatin",
    "lopressor": "metoprolol",
    "glucophage": "metformin",
    "toprol": "metoprolol",
    "norvasc": "amlodipine",
    "prinivil": "lisinopril",
    "zestril": "lisinopril",
}

_STRIP_SUFFIXES = re.compile(
    r"\b(hcl|sodium|sulfate|bisulfate|tartrate|succinate|citrate|besylate|"
    r"phosphate|acetate|maleate|mesylate|ec|xl|sr|er|cr|iv|po|liquid|"
    r"\(liquid\)|\(glass bottle\)|flush|injection|oral|tablet|capsule)\b",
    re.IGNORECASE,
)


def normalize_medication_name(raw_name: str) -> str:
    if not raw_name or not isinstance(raw_name, str):
        return ""
    name = raw_name.strip().lower()
    name = re.sub(r"\d+(\.\d+)?\s*(mg|mcg|g|ml|meq|unit|units|%)\b", "", name)
    name = _STRIP_SUFFIXES.sub("", name)
    name = re.sub(r"[^a-z\s\-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = _SYNONYMS.get(name, name)
    return name


# --- class mapping (exploratory, high-volume drugs only) ---------------
# category -> is_antihypertensive
_ANTIHTN_CLASSES = {
    "ace inhibitor", "arb", "beta blocker", "calcium channel blocker",
    "loop diuretic", "thiazide diuretic", "central alpha agonist",
    "direct vasodilator",
}

_EXCLUDE_ENTIRELY = {
    "d5w", "ns", "sw", "lr", "potassium chloride", "magnesium",
    "sodium chloride", "calcium gluconate", "iso-osmotic dextrose",
    "syringe", "vial", "sterile water",
}

_CLASS_MAP = {
    "metoprolol": "beta blocker",
    "atenolol": "beta blocker",
    "labetalol": "beta blocker",
    "carvedilol": "beta blocker",
    "propranolol": "beta blocker",
    "furosemide": "loop diuretic",
    "bumetanide": "loop diuretic",
    "hydrochlorothiazide": "thiazide diuretic",
    "chlorthalidone": "thiazide diuretic",
    "lisinopril": "ace inhibitor",
    "captopril": "ace inhibitor",
    "enalapril": "ace inhibitor",
    "ramipril": "ace inhibitor",
    "losartan": "arb",
    "valsartan": "arb",
    "amlodipine": "calcium channel blocker",
    "diltiazem": "calcium channel blocker",
    "nifedipine": "calcium channel blocker",
    "verapamil": "calcium channel blocker",
    "clonidine": "central alpha agonist",
    "hydralazine": "direct vasodilator",
    "nitroprusside": "direct vasodilator",
    "acetaminophen": "analgesic - non-opioid",
    "oxycodone-acetaminophen": "analgesic - opioid combination",
    "morphine": "analgesic - opioid",
    "hydromorphone": "analgesic - opioid",
    "oxycodone": "analgesic - opioid",
    "fentanyl citrate": "analgesic - opioid",
    "meperidine": "analgesic - opioid",
    "ketorolac": "nsaid",
    "insulin": "antidiabetic - insulin",
    "metformin": "antidiabetic - biguanide",
    "glipizide": "antidiabetic - sulfonylurea",
    "glyburide": "antidiabetic - sulfonylurea",
    "docusate": "laxative / stool softener",
    "bisacodyl": "laxative / stool softener",
    "senna": "laxative / stool softener",
    "milk of magnesia": "laxative / stool softener",
    "aspirin": "antiplatelet",
    "aspirin ec": "antiplatelet",
    "clopidogrel": "antiplatelet",
    "warfarin": "anticoagulant",
    "heparin": "anticoagulant",
    "atorvastatin": "statin",
    "simvastatin": "statin",
    "pravastatin": "statin",
    "rosuvastatin": "statin",
    "ranitidine": "h2 blocker",
    "famotidine": "h2 blocker",
    "pantoprazole": "proton pump inhibitor",
    "omeprazole": "proton pump inhibitor",
    "lorazepam": "benzodiazepine",
    "midazolam": "benzodiazepine",
    "zolpidem": "sedative - hypnotic",
    "vancomycin": "antibiotic - glycopeptide",
    "levofloxacin": "antibiotic - fluoroquinolone",
    "cefazolin": "antibiotic - cephalosporin",
    "ceftriaxone": "antibiotic - cephalosporin",
    "metronidazole": "antibiotic - other",
    "azithromycin": "antibiotic - macrolide",
    "piperacillin-tazobactam": "antibiotic - penicillin combination",
    "amiodarone": "antiarrhythmic",
    "nitroglycerin": "antianginal - nitrate",
    "propofol": "sedative - anesthetic",
    "metoclopramide": "antiemetic / prokinetic",
    "ondansetron": "antiemetic",
    "haloperidol": "antipsychotic",
    "phenytoin": "anticonvulsant",
    "glycopyrrolate": "anticholinergic",
    "neostigmine": "cholinergic / reversal agent",
    "sucralfate": "gi mucosal protectant",
    "phenylephrine": "vasopressor",
}


def map_to_medication_class(normalized_name: str) -> Optional[str]:
    if not normalized_name:
        return None
    if normalized_name in _EXCLUDE_ENTIRELY:
        return "__EXCLUDE__"
    if normalized_name in _CLASS_MAP:
        return _CLASS_MAP[normalized_name]
    for key, cls in _CLASS_MAP.items():
        if key in normalized_name:
            return cls
    for k in _EXCLUDE_ENTIRELY:
        if k in normalized_name:
            return "__EXCLUDE__"
    return "Other/Unmapped"


def is_antihypertensive_class(cls: str) -> bool:
    return cls in _ANTIHTN_CLASSES

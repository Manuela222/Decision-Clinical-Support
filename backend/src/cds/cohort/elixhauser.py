"""
Elixhauser comorbidity grouping for ICD-9-CM codes, implementing the
published coding algorithm of Quan et al. 2005 ("Coding Algorithms for
Defining Comorbidities in ICD-9-CM and ICD-10 Administrative Data",
Med Care 43(11):1130-9), as commonly re-implemented (e.g. AHRQ Elixhauser
Comorbidity Software, R `comorbidity` package).

Carried forward from the Phase 0 feasibility audit (see
`phase0_feasibility/elixhauser.py` and `FEASIBILITY_REPORT.md`), where it was
flagged as reconstructed from the published literature rather than copied
from a validated package, and NOT yet cross-validated row-for-row against a
reference implementation. That caveat still applies here: this is the
comorbidity grouping scheme `select_cohort()` uses, but it should be
revalidated (e.g. diffed against the `comorbidity` R package or AHRQ's
software on the same MIMIC-III extract) before results are treated as
clinically authoritative.

Quan's algorithm defines two separate categories, "Hypertension,
uncomplicated" (401.x) and "Hypertension, complicated" (402-405.x, the codes
shared with CHF/renal failure). Per the original Elixhauser (1998)
formulation and this project's framing of hypertension as one
condition/comorbidity, both are merged into a single "Hypertension" group.
Codes shared between Hypertension-complicated and Congestive Heart Failure /
Renal Failure are counted toward BOTH groups if present, consistent with the
original algorithm's design (a patient can have both conditions).

MIMIC-III stores ICD9_CODE without decimal points (e.g. "4019" for 401.9,
"5849" for 584.9). All matching below is done against undotted codes.
"""

import re
from typing import Dict, List, Set

# Each entry: category name -> list of code-matching rules, matched via
# string-prefix (`startswith`), which is the standard simplification used for
# administrative/billing data.

ELIXHAUSER_ICD9_PREFIXES: Dict[str, List[str]] = {
    "Congestive heart failure": [
        "39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413",
        "40491", "40493", "428",
    ],
    "Cardiac arrhythmias": [
        "4260", "42613", "4267", "4269", "42610", "42612", "4270", "4272",
        "42731", "42760", "4279", "7850", "99601", "99604", "V450", "V533",
    ],
    "Valvular disease": [
        "0932", "394", "395", "396", "397", "424", "7463", "7464", "7465",
        "7466", "V422", "V433",
    ],
    "Pulmonary circulation disorders": [
        "4150", "4151", "416", "4170", "4178", "4179",
    ],
    "Peripheral vascular disorders": [
        "0930", "4373", "440", "441", "4431", "4432", "4433", "4434", "4435",
        "4436", "4437", "4438", "4439", "4471", "5571", "5579", "V434",
    ],
    "Hypertension": [
        # uncomplicated
        "4011", "4019",
        # complicated (shared w/ CHF, renal failure, etc. per Quan 2005)
        "4010", "402", "403", "404", "405",
    ],
    "Paralysis": [
        "342", "343", "3440", "3441", "3442", "3443", "3444", "3445", "3446",
        "3449",
    ],
    "Other neurological disorders": [
        "3319", "3320", "3334", "3335", "334", "335", "340", "3411", "3412",
        "3413", "3414", "3415", "3416", "3417", "3418", "3419", "345", "3481",
        "3483", "7803", "7843",
    ],
    "Chronic pulmonary disease": [
        "490", "491", "492", "493", "494", "495", "496", "497", "498", "499",
        "500", "501", "502", "503", "504", "505", "5064", "5081", "5088",
    ],
    "Diabetes, uncomplicated": ["2500", "2501", "2502", "2503"],
    "Diabetes, complicated": ["2504", "2505", "2506", "2507", "2508", "2509"],
    "Hypothyroidism": ["2409", "244", "2461", "2468"],
    "Renal failure": [
        "40301", "40311", "40391", "40402", "40403", "40412", "40413",
        "40492", "40493", "585", "586", "V420", "V451", "V56",
    ],
    "Liver disease": [
        "07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709",
        "4560", "4561", "4562", "570", "571", "5722", "5723", "5724", "5725",
        "5726", "5727", "5728", "5733", "5734", "5738", "5739", "V427",
    ],
    "Peptic ulcer disease excl. bleeding": [
        "5317", "5319", "5327", "5329", "5337", "5339", "5347", "5349",
    ],
    "AIDS/HIV": ["042", "043", "044"],
    "Lymphoma": ["200", "201", "202", "2030", "2386"],
    "Metastatic cancer": ["196", "197", "198", "199"],
    "Rheumatoid arthritis/collagen vascular diseases": [
        "446", "7010", "7100", "7101", "7102", "7103", "7104", "7108", "7109",
        "714", "720", "725",
    ],
    "Coagulopathy": ["286", "2871", "2873", "2874", "2875"],
    "Obesity": ["2780"],
    "Weight loss": ["260", "261", "262", "263"],
    "Fluid and electrolyte disorders": ["276"],
    "Blood loss anemia": ["2800"],
    "Deficiency anemia": ["2801", "2802", "2803", "2804", "2805", "2806", "2807", "2808", "2809", "281"],
    "Alcohol abuse": ["2911", "2912", "2913", "2915", "2918", "2919", "3039", "3050", "V113"],
    "Drug abuse": ["292", "304", "3052", "3053", "3054", "3055", "3056", "3057", "3058", "3059"],
    "Psychoses": ["295", "296", "297", "298", "2991"],
    "Depression": ["3004", "30112", "3090", "3091", "311"],
}

# "Solid tumor without metastasis" needs numeric-range precision (140-172,
# 174-195) rather than crude string prefixes, which would over-match.
_SOLID_TUMOR_RANGES = [(140, 172), (174, 195)]


def _code3(icd9_code: str) -> int:
    m = re.match(r"^(\d{3})", icd9_code)
    return int(m.group(1)) if m else -1


def map_icd9_to_elixhauser(icd9_code: str) -> Set[str]:
    """Return the set of Elixhauser categories a single ICD9 code belongs to.

    Most codes map to zero or one category; a handful of codes shared
    between Hypertension-complicated and CHF/renal failure map to two,
    by design of the source algorithm.
    """
    if not icd9_code or not isinstance(icd9_code, str):
        return set()
    code = icd9_code.strip().upper().replace(".", "")
    if not code:
        return set()

    categories: Set[str] = set()
    for category, prefixes in ELIXHAUSER_ICD9_PREFIXES.items():
        for prefix in prefixes:
            if code.startswith(prefix):
                categories.add(category)
                break

    c3 = _code3(code)
    if c3 >= 0 and not code.startswith("V"):
        for lo, hi in _SOLID_TUMOR_RANGES:
            if lo <= c3 <= hi:
                categories.add("Solid tumor without metastasis")
                break

    return categories


ALL_CATEGORIES = sorted(set(ELIXHAUSER_ICD9_PREFIXES.keys()) | {"Solid tumor without metastasis"})

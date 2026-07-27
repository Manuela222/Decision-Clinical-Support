// Static reference data for populating <select> dropdowns only. The
// backend (cds.medications / cds.cohort.elixhauser) is still the source of
// truth for validation -- /predict/agent/new-patient rejects any
// medication class not in its own vocabulary regardless of what this list
// contains. Keeping this list in sync when the backend dictionary changes
// is a manual step for this prototype (see README).
const MEDICATION_CLASSES = [
  "ace inhibitor", "alpha-1 blocker", "analgesic - non-opioid", "analgesic - opioid",
  "analgesic - opioid combination", "antianginal - nitrate", "antiarrhythmic",
  "antibiotic - aminoglycoside", "antibiotic - carbapenem", "antibiotic - cephalosporin",
  "antibiotic - fluoroquinolone", "antibiotic - glycopeptide", "antibiotic - macrolide",
  "antibiotic - other", "antibiotic - penicillin", "antibiotic - penicillin combination",
  "anticholinergic", "anticoagulant", "anticonvulsant", "anticonvulsant / neuropathic pain",
  "antidepressant", "antidiabetic - biguanide", "antidiabetic - insulin",
  "antidiabetic - sulfonylurea", "antidiabetic - thiazolidinedione", "antiemetic",
  "antiemetic / prokinetic", "antifungal", "antihistamine", "antihypoglycemic agent",
  "antiplatelet", "antipsychotic", "antiviral", "anxiolytic - non-benzodiazepine", "arb",
  "benzodiazepine", "beta blocker", "bronchodilator - anticholinergic",
  "bronchodilator - beta agonist", "bronchodilator - combination", "calcium channel blocker",
  "calcium supplement", "central alpha agonist", "cholinergic / reversal agent",
  "cholinesterase inhibitor", "corticosteroid - systemic", "decongestant",
  "direct vasodilator", "expectorant", "gi antiflatulent", "gi mucosal protectant",
  "gout medication", "h2 blocker", "heparin reversal agent", "inhaled corticosteroid",
  "inhaled corticosteroid combination", "iron supplement", "laxative / stool softener",
  "leukotriene modifier", "lipid-lowering - non-statin", "local anesthetic", "loop diuretic",
  "mineralocorticoid", "neuromuscular blocker", "nsaid", "phosphate binder",
  "potassium binder", "proton pump inhibitor", "sedative - anesthetic", "sedative - hypnotic",
  "statin", "thiazide diuretic", "thrombolytic", "thyroid hormone replacement", "vasopressor",
  "vitamin / supplement", "vitamin k / anticoagulation reversal",
];

const ADMISSION_REASONS = [
  "AIDS/HIV", "Alcohol abuse", "Blood loss anemia", "Cardiac arrhythmias",
  "Chronic pulmonary disease", "Coagulopathy", "Congestive heart failure",
  "Deficiency anemia", "Depression", "Diabetes, complicated", "Diabetes, uncomplicated",
  "Drug abuse", "Fluid and electrolyte disorders", "Hypothyroidism", "Liver disease",
  "Lymphoma", "Metastatic cancer", "Obesity", "Other neurological disorders", "Paralysis",
  "Peptic ulcer disease excl. bleeding", "Peripheral vascular disorders", "Psychoses",
  "Pulmonary circulation disorders", "Renal failure",
  "Rheumatoid arthritis/collagen vascular diseases", "Solid tumor without metastasis",
  "Valvular disease", "Weight loss",
];

const HYPERTENSION_STATUSES = [
  { value: "confirmed_chronic", label: "Confirmed chronic hypertension" },
  { value: "suspected", label: "Suspected hypertension" },
  { value: "not_present", label: "No hypertension" },
  { value: "unknown", label: "Unknown" },
];

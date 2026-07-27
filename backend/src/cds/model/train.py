"""Phase 9: train the multi-label discharge-medication-class classifier.

TEXT EMBEDDING CHOICE (documented per the project spec's request to justify
this): note text is embedded with TF-IDF, not a pretrained clinical
language model (e.g. ClinicalBERT/BioBERT). Reasons, in order of weight:
  1. This prototype's training split is small (see Phase 0 audit: on the
     order of ~500 admissions) — a pretrained transformer embedding adds
     substantial complexity and a large new dependency (torch/transformers,
     GPU-friendly or not) for a dataset too small to benefit much from it
     over a well-regularized linear/tree model on TF-IDF features.
  2. TF-IDF is fully deterministic, requires no model download, and keeps
     the dependency footprint to scikit-learn (already required for the
     classifier), which matters for a prototype meant to run offline.
  3. Feature importances from TF-IDF terms are directly interpretable for
     Phase 15's explainability report; pretrained embedding dimensions are
     not.
If this were scaled up to full MIMIC-III (~4-5x more admissions per the
Phase 0 audit's extrapolation), revisiting this with a pretrained clinical
embedding model would be worth it and should be reconsidered then.

MODEL CHOICE: RandomForestClassifier (scikit-learn) instead of XGBoost —
it supports multi-label targets (a 2D binary indicator matrix) natively
with no extra wrapper, ships with scikit-learn (already a dependency, no
new heavy dependency), and exposes `feature_importances_` directly for
Phase 15, at the cost of somewhat weaker raw performance than a
well-tuned XGBoost model — an acceptable trade for this prototype's scope.
"""
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from ..evaluation import compute_multilabel_metrics
from ..schemas import ClinicalState
from .features import ALL_FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE, build_structured_features
from .model_card import ModelCard

DEFAULT_MODEL_VERSION = "rf-tfidf-v1"

KNOWN_LIMITATIONS = [
    "Trained on a small cohort (see model card's n_training_admissions); metrics are not statistically robust.",
    "Text embeddings are TF-IDF, not a pretrained clinical language model — see train.py module docstring for why.",
    "Antihypertensive medication classes are excluded from the label space by convention (see cds.medications), not learned.",
    "No blood pressure feature is available (see cds.timeline.state_builder docstring, point 3); hypertension "
    "signal comes only from diagnosis codes and antihypertensive medication presence.",
    "The internal validation split used for this model card is admission-level, not patient-level — only the "
    "official Phase 6 train/test split (used for Phase 14 evaluation) is guaranteed patient-disjoint.",
]


@dataclass
class TrainedModelArtifact:
    pipeline: Pipeline
    label_binarizer: MultiLabelBinarizer
    model_card: ModelCard


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("text", TfidfVectorizer(max_features=300, min_df=1), TEXT_FEATURE),
        ]
    )
    classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    return Pipeline([("preprocess", preprocessor), ("classify", classifier)])


def _binary_matrix_to_classes(y_matrix, classes) -> List[List[str]]:
    return [[classes[i] for i, flag in enumerate(row) if flag] for row in y_matrix]


def train_model(
    clinical_states: List[ClinicalState],
    ground_truth_classes: List[List[str]],
    model_version: str = DEFAULT_MODEL_VERSION,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> TrainedModelArtifact:
    """Train on `clinical_states`/`ground_truth_classes` (the Phase 6 TRAIN
    split only — never pass test-split data here). Internally carves out a
    validation subset (seeded, admission-level) purely to report metrics in
    the model card, then refits on the full training input for the
    artifact actually used downstream."""
    if len(clinical_states) != len(ground_truth_classes):
        raise ValueError(
            f"clinical_states ({len(clinical_states)}) and ground_truth_classes "
            f"({len(ground_truth_classes)}) must be the same length."
        )
    if len(clinical_states) < 5:
        raise ValueError("Need at least 5 training admissions to fit and internally validate a model.")

    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(ground_truth_classes)
    features_all = build_structured_features(clinical_states)

    indices = list(range(len(clinical_states)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, round(len(indices) * validation_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    validation_pipeline = build_pipeline()
    validation_pipeline.fit(features_all.iloc[train_idx][ALL_FEATURE_COLUMNS], y_all[train_idx])
    y_val_pred = validation_pipeline.predict(features_all.iloc[val_idx][ALL_FEATURE_COLUMNS])
    val_pred_classes = _binary_matrix_to_classes(y_val_pred, mlb.classes_)
    val_true_classes = [ground_truth_classes[i] for i in val_idx]
    validation_metrics = compute_multilabel_metrics(val_true_classes, val_pred_classes)

    # Refit on the full training input for the artifact used downstream, now
    # that validation metrics have been captured on a held-out subset.
    final_pipeline = build_pipeline()
    final_pipeline.fit(features_all[ALL_FEATURE_COLUMNS], y_all)

    model_card = ModelCard(
        model_version=model_version,
        trained_at=datetime.now(timezone.utc),
        n_training_admissions=len(clinical_states),
        n_validation_admissions=len(val_idx),
        feature_columns=list(ALL_FEATURE_COLUMNS),
        label_classes=list(mlb.classes_),
        text_embedding_method=(
            "TF-IDF (max_features=300, unigrams) over truncated note excerpts — "
            "see train.py module docstring for why not a pretrained clinical embedding model."
        ),
        classifier_type="scikit-learn RandomForestClassifier(n_estimators=200, class_weight='balanced'), natively multi-label",
        validation_metrics={k: v for k, v in validation_metrics.items() if k != "per_admission"},
        known_limitations=list(KNOWN_LIMITATIONS),
    )

    return TrainedModelArtifact(pipeline=final_pipeline, label_binarizer=mlb, model_card=model_card)

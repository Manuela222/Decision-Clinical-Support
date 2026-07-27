"""Shared profile encoder for the RAG index: the same structured-feature
extraction Phase 9 uses (age, gender, admission condition, hypertension
status, renal labs, medication counts, note text), vectorized for
similarity search rather than classification. No trained classifier is
reused here — only the label-agnostic feature extraction utility."""
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..model.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE


def build_profile_encoder() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("text", TfidfVectorizer(max_features=200, min_df=1), TEXT_FEATURE),
        ]
    )

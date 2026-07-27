"""Global feature importances from the trained model's RandomForestClassifier,
mapped back through the ColumnTransformer's output feature names."""
from typing import List

from ..model import TrainedModelArtifact
from .schemas import FeatureImportance


def get_feature_importances(artifact: TrainedModelArtifact, top_n: int = 15) -> List[FeatureImportance]:
    preprocessor = artifact.pipeline.named_steps["preprocess"]
    classifier = artifact.pipeline.named_steps["classify"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    ranked = sorted(zip(feature_names, importances), key=lambda pair: -pair[1])
    return [FeatureImportance(feature=str(name), importance=float(value)) for name, value in ranked[:top_n]]

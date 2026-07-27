"""ModelCard: a JSON-serializable record of what the trained model is, how
it was trained, and what it's known not to be good for."""
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel


class ModelCard(BaseModel):
    model_version: str
    trained_at: datetime
    n_training_admissions: int
    n_validation_admissions: int
    feature_columns: List[str]
    label_classes: List[str]
    text_embedding_method: str
    classifier_type: str
    validation_metrics: Dict[str, Any]
    known_limitations: List[str]

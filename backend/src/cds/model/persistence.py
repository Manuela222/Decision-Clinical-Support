"""Save/load the trained model artifact (Phase 9 deliverable: a saved
model artifact plus its model card)."""
from pathlib import Path

import joblib

from .model_card import ModelCard
from .train import TrainedModelArtifact


def save_model_artifact(artifact: TrainedModelArtifact, path: "Path | str") -> None:
    """Persist the pipeline + label binarizer (joblib, `.joblib`) and write
    the model card alongside it as human-readable JSON (`<path>.model_card.json`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    card_path = path.with_suffix(path.suffix + ".model_card.json")
    card_path.write_text(artifact.model_card.model_dump_json(indent=2))


def load_model_artifact(path: "Path | str") -> TrainedModelArtifact:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No trained model artifact found at '{path}'.")
    return joblib.load(path)


def load_model_card(path: "Path | str") -> ModelCard:
    """Load just the human-readable model card JSON without unpickling the model."""
    path = Path(path)
    card_path = path.with_suffix(path.suffix + ".model_card.json")
    if not card_path.is_file():
        raise FileNotFoundError(f"No model card found at '{card_path}'.")
    return ModelCard.model_validate_json(card_path.read_text())

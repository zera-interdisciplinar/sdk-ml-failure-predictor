import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from src.data.features import CATEGORICAL_COLUMNS, EncodedRecord, FeatureEncoder
from src.data.schema import DeviceRecord
from src.models.ttf_model import TTFModel
from src.serving.schema import PredictionError, PredictionRequest

ROOT: Path = Path(__file__).resolve().parents[2]


def bundle_dir() -> Path:
    raw = os.environ.get("MODEL_DIR", "checkpoints/serving")
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


class TTFPredictor:
    """Loads the trained model and feature encoder once, then answers
    prediction requests without touching disk again."""

    def __init__(self, model_dir: Path | None = None) -> None:
        directory: Path = model_dir if model_dir is not None else bundle_dir()
        manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text())
        self.encoder: FeatureEncoder = FeatureEncoder.load(directory / "feature_encoder.json")

        vocab_sizes: dict[str, int] = {col: self.encoder.vocab_size(col) for col in CATEGORICAL_COLUMNS}
        if vocab_sizes != manifest["vocab_sizes"]:
            raise RuntimeError(
                f"encoder vocab {vocab_sizes} does not match manifest {manifest['vocab_sizes']}"
            )

        checkpoint: Path = directory / "ttf_model.pt"
        digest: str = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != manifest["state_dict_sha256"]:
            raise RuntimeError("checkpoint sha256 does not match manifest")

        self.model: TTFModel = TTFModel(
            vocab_sizes=vocab_sizes,
            embedding_dims=manifest["embedding_dims"],
            hidden_layers=manifest["hidden_layers"],
        )
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, payload: dict[str, Any]) -> float:
        return self.predict_batch([payload])[0]

    def predict_batch(self, payloads: list[dict[str, Any]]) -> list[float]:
        """Runs every payload through the model as a single batched forward
        pass (one tensor with N rows), instead of N separate forward passes —
        the cost of a forward pass barely grows with N, so batching many
        devices together is far cheaper per-item than calling predict() N
        times."""
        records: list[DeviceRecord] = [DeviceRecord.model_validate(p) for p in payloads]
        encoded: list[EncodedRecord] = [self.encoder.transform(r) for r in records]

        categorical: torch.Tensor = torch.tensor(
            [[enc["categorical"][col] for col in CATEGORICAL_COLUMNS] for enc in encoded], dtype=torch.long
        )
        numeric: torch.Tensor = torch.tensor([enc["numeric"] for enc in encoded], dtype=torch.float32)

        with torch.no_grad():
            predictions: torch.Tensor = self.model(categorical, numeric)

        return [max(0.0, float(value)) for value in predictions.tolist()]

    def predict_batch_validated(self, requests: list[dict[str, Any]]) -> list[float | PredictionError]:
        """Same as predict_batch, but validates each item independently
        first: an invalid item (missing/mistyped field) becomes a
        PredictionError at its position instead of failing the whole call.
        Shared by the MCP tool and the REST endpoint so validation stays
        in one place."""
        records: dict[int, dict[str, Any]] = {}
        results: list[float | PredictionError] = [PredictionError(error="not processed") for _ in requests]

        for i, raw in enumerate(requests):
            try:
                validated: PredictionRequest = PredictionRequest.model_validate(raw)
            except ValidationError as exc:
                results[i] = PredictionError(error=str(exc))
                continue
            records[i] = validated.model_dump(mode="json")

        if records:
            predictions: list[float] = self.predict_batch(list(records.values()))
            for i, prediction in zip(records.keys(), predictions):
                results[i] = prediction

        return results

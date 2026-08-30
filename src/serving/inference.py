from pathlib import Path
from typing import Any

import torch
import yaml

from src.data.features import CATEGORICAL_COLUMNS, EncodedRecord, FeatureEncoder
from src.data.schema import DeviceRecord
from src.models.ttf_model import TTFModel

CONFIG_PATH: Path = Path(__file__).parent.parent / "training" / "config.yaml"


class TTFPredictor:
    """Loads the trained model and feature encoder once, then answers
    prediction requests without touching disk again."""

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        config: dict[str, Any] = yaml.safe_load(Path(config_path).read_text())

        self.encoder: FeatureEncoder = FeatureEncoder.load(config["data"]["encoder_path"])

        vocab_sizes: dict[str, int] = {col: self.encoder.vocab_size(col) for col in CATEGORICAL_COLUMNS}
        self.model: TTFModel = TTFModel(
            vocab_sizes=vocab_sizes,
            embedding_dims=config["model"]["embedding_dims"],
            hidden_layers=config["model"]["hidden_layers"],
        )
        self.model.load_state_dict(torch.load(config["model"]["checkpoint_path"], map_location="cpu"))
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

        return predictions.tolist()

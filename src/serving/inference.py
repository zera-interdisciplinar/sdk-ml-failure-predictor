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
        record: DeviceRecord = DeviceRecord.model_validate(payload)
        encoded: EncodedRecord = self.encoder.transform(record)

        categorical: torch.Tensor = torch.tensor(
            [[encoded["categorical"][col] for col in CATEGORICAL_COLUMNS]], dtype=torch.long
        )
        numeric: torch.Tensor = torch.tensor([encoded["numeric"]], dtype=torch.float32)

        with torch.no_grad():
            prediction: torch.Tensor = self.model(categorical, numeric)

        return prediction.item()

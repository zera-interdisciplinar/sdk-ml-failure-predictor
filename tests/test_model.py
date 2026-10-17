import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
import torch

from src.data.features import CATEGORICAL_COLUMNS, FeatureEncoder, months_since_manufacturing_year
from src.data.schema import DeviceRecord
from src.models.ttf_model import TTFModel
from src.serving.inference import ROOT, TTFPredictor, bundle_dir
from src.serving.schema import PredictionError


def test_age_counts_months_inside_the_manufacturing_year() -> None:
    assert months_since_manufacturing_year(date(2024, 1, 1), 2024) == 0
    july = months_since_manufacturing_year(date(2024, 7, 1), 2024)
    assert july == pytest.approx(182 / 30.44, rel=1e-6)
    assert july > 5


def _record(model: str, acquired: date) -> DeviceRecord:
    return DeviceRecord(
        category="notebook",
        manufacturer="Dell",
        model=model,
        climateZone="TROPICAL",
        usageIntensity=5,
        manufacturingDate=2022,
        acquiredAt=acquired,
        mesesAteQuebra=40.0,
    )


def _write_bundle(directory: Path, digest: str | None = None, vocab_sizes: dict[str, int] | None = None) -> None:
    records = [_record("Latitude 5420", date(2023, 1, 15)), _record("XPS 13", date(2023, 8, 1))]
    encoder = FeatureEncoder()
    encoder.fit(records)
    encoder.save(directory / "feature_encoder.json")
    sizes = {col: encoder.vocab_size(col) for col in CATEGORICAL_COLUMNS}
    dims = {col: 2 for col in CATEGORICAL_COLUMNS}
    model = TTFModel(sizes, dims, [4])
    checkpoint = directory / "ttf_model.pt"
    torch.save(model.state_dict(), checkpoint)
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "embedding_dims": dims,
        "hidden_layers": [4],
        "vocab_sizes": vocab_sizes or sizes,
        "seed": 1,
        "best_epoch": 1,
        "val_mae": 1.0,
        "val_rmse": 1.0,
        "unseen_model_mae": 1.0,
        "unseen_model_rmse": 1.0,
        "state_dict_sha256": digest or actual,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_bundle_dir_is_anchored_at_the_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_DIR", "checkpoints/serving")
    assert bundle_dir() == ROOT / "checkpoints/serving"


def test_prediction_is_never_negative(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    predictor = TTFPredictor(tmp_path)

    def negative(_categorical: torch.Tensor, _numeric: torch.Tensor) -> torch.Tensor:
        return torch.tensor([-4.2, 3.5])

    predictor.model.forward = negative
    results = predictor.predict_batch(
        [
            _record("Latitude 5420", date(2023, 1, 15)).model_dump(mode="json"),
            _record("no-such-model", date(2024, 6, 1)).model_dump(mode="json"),
        ]
    )
    assert results == [0.0, 3.5]


def test_manifest_mismatch_refuses_to_start(tmp_path: Path) -> None:
    _write_bundle(tmp_path, digest="0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        TTFPredictor(tmp_path)


@pytest.mark.skipif(
    not (ROOT / "checkpoints" / "serving" / "manifest.json").is_file(),
    reason="o servidor carrega checkpoints/serving, que o treino ainda vai gravar",
)
def test_invalid_json_is_422() -> None:
    from src.serving.mcp_server import predict_batch_http

    class FakeRequest:
        async def json(self) -> dict:
            raise json.JSONDecodeError("bad", "", 0)

    import asyncio

    response = asyncio.run(predict_batch_http(FakeRequest()))
    assert response.status_code == 422


def test_invalid_batch_item_does_not_drop_the_rest(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    predictor = TTFPredictor(tmp_path)
    results = predictor.predict_batch_validated(
        [
            {"category": "notebook"},
            _record("Latitude 5420", date(2023, 1, 15)).model_dump(mode="json"),
        ]
    )
    assert isinstance(results[0], PredictionError)
    assert isinstance(results[1], float)
    assert results[1] >= 0

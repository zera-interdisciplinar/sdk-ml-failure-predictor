import json
from pathlib import Path

from src.data.schema import DeviceRecord

CATEGORICAL_COLUMNS = ["category", "manufacturer", "model", "climateZone"]
NUMERIC_COLUMNS = ["usageIntensity", "ageAtAcquisitionMonths"]

UNK_TOKEN = "<UNK>"


def age_at_acquisition_months(record: DeviceRecord) -> float:
    return max((record.acquiredAt.year - record.manufacturingDate) * 12, 0)


def target_months(record: DeviceRecord) -> float:
    if record.mesesAteQuebra is not None:
        return record.mesesAteQuebra
    if record.actualBreakDate is not None:
        delta_days = (record.actualBreakDate - record.acquiredAt).days
        return delta_days / 30.44
    raise ValueError("record has neither mesesAteQuebra nor actualBreakDate to derive the target")


class FeatureEncoder:
    """Fits categorical vocabularies and numeric normalization stats on the
    training set, then encodes records the same way at train and serve time.
    Must be fit once, saved, and loaded (not refit) wherever it's reused, so
    both sides apply the identical mapping."""

    def __init__(self) -> None:
        self.vocabs: dict[str, dict[str, int]] = {}
        self.numeric_mean: dict[str, float] = {}
        self.numeric_std: dict[str, float] = {}

    def fit(self, records: list[DeviceRecord]) -> None:
        for col in CATEGORICAL_COLUMNS:
            values = sorted({getattr(r, col) for r in records})
            self.vocabs[col] = {UNK_TOKEN: 0, **{v: i + 1 for i, v in enumerate(values)}}

        numeric_values: dict[str, list[float]] = {col: [] for col in NUMERIC_COLUMNS}
        for r in records:
            numeric_values["usageIntensity"].append(r.usageIntensity)
            numeric_values["ageAtAcquisitionMonths"].append(age_at_acquisition_months(r))

        for col, values in numeric_values.items():
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            self.numeric_mean[col] = mean
            self.numeric_std[col] = variance**0.5 or 1.0

    def encode_categorical(self, record: DeviceRecord) -> dict[str, int]:
        return {
            col: self.vocabs[col].get(getattr(record, col), self.vocabs[col][UNK_TOKEN])
            for col in CATEGORICAL_COLUMNS
        }

    def encode_numeric(self, record: DeviceRecord) -> list[float]:
        raw = {
            "usageIntensity": record.usageIntensity,
            "ageAtAcquisitionMonths": age_at_acquisition_months(record),
        }
        return [(raw[col] - self.numeric_mean[col]) / self.numeric_std[col] for col in NUMERIC_COLUMNS]

    def transform(self, record: DeviceRecord) -> dict:
        return {
            "categorical": self.encode_categorical(record),
            "numeric": self.encode_numeric(record),
        }

    def vocab_size(self, col: str) -> int:
        return len(self.vocabs[col])

    def save(self, path: str | Path) -> None:
        payload = {
            "vocabs": self.vocabs,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEncoder":
        payload = json.loads(Path(path).read_text())
        encoder = cls()
        encoder.vocabs = payload["vocabs"]
        encoder.numeric_mean = payload["numeric_mean"]
        encoder.numeric_std = payload["numeric_std"]
        return encoder


def load_records(path: str | Path) -> list[DeviceRecord]:
    raw = json.loads(Path(path).read_text())
    return [DeviceRecord.model_validate(r) for r in raw]

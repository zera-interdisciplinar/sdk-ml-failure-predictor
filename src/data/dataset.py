import torch
from torch.utils.data import Dataset

from src.data.features import CATEGORICAL_COLUMNS, FeatureEncoder, target_months
from src.data.schema import DeviceRecord


class DeviceDataset(Dataset):
    def __init__(self, records: list[DeviceRecord], encoder: FeatureEncoder) -> None:
        self.records = records
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        encoded = self.encoder.transform(record)


        categorical = torch.tensor(
            [encoded["categorical"][col] for col in CATEGORICAL_COLUMNS], dtype=torch.long
        )
        numeric = torch.tensor(encoded["numeric"], dtype=torch.float32)
        target = torch.tensor(target_months(record), dtype=torch.float32)

        return categorical, numeric, target

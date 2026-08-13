import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.data.dataset import DeviceDataset
from src.data.features import CATEGORICAL_COLUMNS, FeatureEncoder, load_records
from src.data.schema import DeviceRecord
from src.models.ttf_model import TTFModel

CONFIG_PATH: Path = Path(__file__).parent / "config.yaml"


def split_records(
    records: list[DeviceRecord], val_split: float, seed: int
) -> tuple[list[DeviceRecord], list[DeviceRecord]]:
    shuffled: list[DeviceRecord] = records.copy()
    random.Random(seed).shuffle(shuffled)
    val_size: int = int(len(shuffled) * val_split)
    return shuffled[val_size:], shuffled[:val_size]


def run_epoch(
    model: TTFModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training: bool = optimizer is not None
    model.train(is_training)

    total_loss: float = 0.0
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            categorical: torch.Tensor
            numeric: torch.Tensor
            target: torch.Tensor
            categorical, numeric, target = batch

            pred: torch.Tensor = model(categorical, numeric)
            loss: torch.Tensor = criterion(pred, target)

            if is_training:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(target)

    return total_loss / len(loader.dataset)


def plot_loss_curve(train_losses: list[float], val_losses: list[float], out_path: Path) -> None:
    plt.figure()
    plt.plot(train_losses, label="treino")
    plt.plot(val_losses, label="validação")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("Loss por epoch")
    plt.savefig(out_path)
    plt.close()


def plot_embeddings(model: TTFModel, encoder: FeatureEncoder, col: str, out_path: Path) -> None:
    vectors: np.ndarray = model.embeddings[col].weight.detach().numpy()
    labels: list[str] = sorted(encoder.vocabs[col], key=encoder.vocabs[col].get)

    projected: np.ndarray = PCA(n_components=2).fit_transform(vectors)

    plt.figure()
    plt.scatter(projected[:, 0], projected[:, 1])
    for label, (x, y) in zip(labels, projected):
        plt.annotate(label, (x, y))
    plt.title(f"Embedding de '{col}' projetada em 2D")
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    config: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())

    records: list[DeviceRecord] = load_records(config["data"]["raw_path"])
    train_records: list[DeviceRecord]
    val_records: list[DeviceRecord]
    train_records, val_records = split_records(
        records, config["data"]["val_split"], config["data"]["seed"]
    )

    encoder: FeatureEncoder = FeatureEncoder()
    encoder.fit(train_records)
    encoder.save(config["data"]["encoder_path"])

    train_loader: DataLoader = DataLoader(
        DeviceDataset(train_records, encoder),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader: DataLoader = DataLoader(
        DeviceDataset(val_records, encoder),
        batch_size=config["training"]["batch_size"],
    )

    vocab_sizes: dict[str, int] = {col: encoder.vocab_size(col) for col in CATEGORICAL_COLUMNS}
    model: TTFModel = TTFModel(
        vocab_sizes=vocab_sizes,
        embedding_dims=config["model"]["embedding_dims"],
        hidden_layers=config["model"]["hidden_layers"],
    )

    criterion: nn.Module = nn.MSELoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss: float = run_epoch(model, train_loader, criterion, optimizer)
        val_loss: float = run_epoch(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"epoch {epoch:03d} | treino: {train_loss:.2f} | validação: {val_loss:.2f}")

    checkpoint_path: Path = Path(config["model"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    plot_loss_curve(train_losses, val_losses, checkpoint_path.parent / "loss_curve.png")
    plot_embeddings(model, encoder, "manufacturer", checkpoint_path.parent / "embedding_manufacturer.png")
    plot_embeddings(model, encoder, "category", checkpoint_path.parent / "embedding_category.png")


if __name__ == "__main__":
    main()

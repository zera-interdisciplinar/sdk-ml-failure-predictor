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
TOLERANCE_MONTHS: float = 3.0


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


PERCENTILES: list[float] = [0.50, 0.75, 0.90, 0.99]


def compute_metrics(
    model: TTFModel, loader: DataLoader, tolerance_months: float
) -> dict[str, float]:
    model.eval()

    total_abs_error: float = 0.0
    total_sq_error: float = 0.0
    correct: int = 0
    total: int = 0
    abs_errors: list[torch.Tensor] = []

    with torch.no_grad():
        for categorical, numeric, target in loader:
            pred: torch.Tensor = model(categorical, numeric)
            errors: torch.Tensor = pred - target
            abs_error: torch.Tensor = errors.abs()

            total_abs_error += abs_error.sum().item()
            total_sq_error += (errors**2).sum().item()
            correct += (abs_error <= tolerance_months).sum().item()
            total += len(target)
            abs_errors.append(abs_error)

    all_abs_errors: torch.Tensor = torch.cat(abs_errors)
    quantiles: torch.Tensor = torch.quantile(all_abs_errors, torch.tensor(PERCENTILES))

    metrics: dict[str, float] = {
        "mae": total_abs_error / total,
        "rmse": (total_sq_error / total) ** 0.5,
        "accuracy": correct / total,
    }
    for percentile, value in zip(PERCENTILES, quantiles.tolist()):
        metrics[f"p{int(percentile * 100)}"] = value

    return metrics


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

    train_metrics: dict[str, float] = compute_metrics(model, train_loader, TOLERANCE_MONTHS)
    val_metrics: dict[str, float] = compute_metrics(model, val_loader, TOLERANCE_MONTHS)

    def format_metrics(label: str, metrics: dict[str, float]) -> str:
        percentiles: str = " | ".join(
            f"p{int(p * 100)}: {metrics[f'p{int(p * 100)}']:.2f}" for p in PERCENTILES
        )
        return (
            f"{label} -> MAE: {metrics['mae']:.2f} meses | "
            f"RMSE: {metrics['rmse']:.2f} | "
            f"acurácia (±{TOLERANCE_MONTHS:.0f} meses): {metrics['accuracy'] * 100:.1f}% | "
            f"{percentiles}"
        )

    print(format_metrics("treino    ", train_metrics))
    print(format_metrics("validação ", val_metrics))

    checkpoint_path: Path = Path(config["model"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    plot_loss_curve(train_losses, val_losses, checkpoint_path.parent / "loss_curve.png")
    plot_embeddings(model, encoder, "manufacturer", checkpoint_path.parent / "embedding_manufacturer.png")
    plot_embeddings(model, encoder, "category", checkpoint_path.parent / "embedding_category.png")


if __name__ == "__main__":
    main()

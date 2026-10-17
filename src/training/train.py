import hashlib
import json
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
ROOT: Path = Path(__file__).resolve().parents[2]
TOLERANCE_MONTHS: float = 3.0


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def split_records(
    records: list[DeviceRecord], val_split: float, seed: int
) -> tuple[list[DeviceRecord], list[DeviceRecord]]:
    shuffled: list[DeviceRecord] = records.copy()
    random.Random(seed).shuffle(shuffled)
    val_size: int = int(len(shuffled) * val_split)
    return shuffled[val_size:], shuffled[:val_size]


def split_model_holdout(
    records: list[DeviceRecord], holdout: float, seed: int
) -> tuple[list[DeviceRecord], list[DeviceRecord]]:
    """Holds out whole model names so the unseen-model metric is not just
    new rows of a name the embedding already trained on."""
    names: list[str] = sorted({record.model for record in records})
    shuffled_names: list[str] = names.copy()
    random.Random(seed).shuffle(shuffled_names)
    held: set[str] = set(shuffled_names[: int(len(shuffled_names) * holdout)])
    seen: list[DeviceRecord] = [record for record in records if record.model not in held]
    unseen: list[DeviceRecord] = [record for record in records if record.model in held]
    return seen, unseen


def run_epoch(
    model: TTFModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    unk_mask_prob: float = 0.0,
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
            if is_training and unk_mask_prob > 0:
                drop: torch.Tensor = torch.rand(categorical.shape) < unk_mask_prob
                categorical = categorical.masked_fill(drop, 0)

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
    total_ape: float = 0.0
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
            total_ape += (abs_error / target.abs().clamp(min=1e-6)).sum().item()
            correct += (abs_error <= tolerance_months).sum().item()
            total += len(target)
            abs_errors.append(abs_error)

    all_abs_errors: torch.Tensor = torch.cat(abs_errors)
    quantiles: torch.Tensor = torch.quantile(all_abs_errors, torch.tensor(PERCENTILES))

    metrics: dict[str, float] = {
        "mae": total_abs_error / total,
        "rmse": (total_sq_error / total) ** 0.5,
        "mape": total_ape / total * 100,
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


def plot_prediction_error(model: TTFModel, loader: DataLoader, out_path: Path) -> None:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for categorical, numeric, target in loader:
            preds.append(model(categorical, numeric))
            targets.append(target)

    pred_arr: np.ndarray = torch.cat(preds).numpy()
    target_arr: np.ndarray = torch.cat(targets).numpy()
    relative_error: np.ndarray = np.abs(pred_arr - target_arr) / np.clip(target_arr, 1e-6, None) * 100

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_scatter.scatter(target_arr, pred_arr, alpha=0.4, s=12)
    lims: list[float] = [0, max(target_arr.max(), pred_arr.max())]
    ax_scatter.plot(lims, lims, "r--", label="previsão ideal")
    ax_scatter.set_xlabel("meses reais até falha")
    ax_scatter.set_ylabel("meses previstos")
    ax_scatter.set_title("Previsto vs real")
    ax_scatter.legend()

    ax_hist.hist(relative_error, bins=30)
    ax_hist.set_xlabel("erro relativo (%)")
    ax_hist.set_ylabel("contagem")
    ax_hist.set_title("Distribuição do erro relativo")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


RANGE_BINS: list[float] = [0, 6, 12, 24, 48, float("inf")]
RANGE_LABELS: list[str] = ["0-6", "6-12", "12-24", "24-48", "48+"]


def plot_error_by_range(model: TTFModel, loader: DataLoader, out_path: Path) -> None:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for categorical, numeric, target in loader:
            preds.append(model(categorical, numeric))
            targets.append(target)

    pred_arr: np.ndarray = torch.cat(preds).numpy()
    target_arr: np.ndarray = torch.cat(targets).numpy()
    abs_error: np.ndarray = np.abs(pred_arr - target_arr)
    relative_error: np.ndarray = abs_error / np.clip(target_arr, 1e-6, None) * 100

    bucket_idx: np.ndarray = np.digitize(target_arr, RANGE_BINS[1:-1])

    mae_per_bucket: list[float] = []
    mape_per_bucket: list[float] = []
    counts: list[int] = []
    for i in range(len(RANGE_LABELS)):
        mask: np.ndarray = bucket_idx == i
        counts.append(int(mask.sum()))
        mae_per_bucket.append(float(abs_error[mask].mean()) if mask.any() else 0.0)
        mape_per_bucket.append(float(relative_error[mask].mean()) if mask.any() else 0.0)

    fig, (ax_mae, ax_mape) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(RANGE_LABELS))
    ax_mae.bar(x, mae_per_bucket)
    ax_mae.set_xticks(x, [f"{lbl}\n(n={n})" for lbl, n in zip(RANGE_LABELS, counts)])
    ax_mae.set_xlabel("faixa do tempo real até falha (meses)")
    ax_mae.set_ylabel("erro absoluto médio (meses)")
    ax_mae.set_title("Erro absoluto por faixa")

    ax_mape.bar(x, mape_per_bucket, color="orange")
    ax_mape.set_xticks(x, [f"{lbl}\n(n={n})" for lbl, n in zip(RANGE_LABELS, counts)])
    ax_mape.set_xlabel("faixa do tempo real até falha (meses)")
    ax_mape.set_ylabel("erro relativo médio (%)")
    ax_mape.set_title("Erro relativo por faixa")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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


def format_metrics(label: str, metrics: dict[str, float]) -> str:
    percentiles: str = " | ".join(
        f"p{int(p * 100)}: {metrics[f'p{int(p * 100)}']:.2f}" for p in PERCENTILES
    )
    return (
        f"{label} -> MAE: {metrics['mae']:.2f} meses | "
        f"RMSE: {metrics['rmse']:.2f} | "
        f"acurácia (±{TOLERANCE_MONTHS:.0f} meses): {metrics['accuracy'] * 100:.1f}% | "
        f"MAPE: {metrics['mape']:.1f}% | "
        f"{percentiles}"
    )


def main() -> None:
    config: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())
    seed: int = config["data"]["seed"]
    torch.manual_seed(seed)

    records: list[DeviceRecord] = load_records(resolve(config["data"]["raw_path"]))
    seen_records, unseen_records = split_model_holdout(records, config["data"]["model_holdout"], seed)
    train_records, val_records = split_records(seen_records, config["data"]["val_split"], seed)

    bundle_dir: Path = resolve(config["model"]["bundle_dir"])
    bundle_dir.mkdir(parents=True, exist_ok=True)
    encoder: FeatureEncoder = FeatureEncoder()
    encoder.fit(train_records)
    encoder_path: Path = bundle_dir / "feature_encoder.json"
    encoder.save(encoder_path)

    batch_size: int = config["training"]["batch_size"]
    loader_gen: torch.Generator = torch.Generator()
    loader_gen.manual_seed(seed)
    train_loader: DataLoader = DataLoader(
        DeviceDataset(train_records, encoder),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_gen,
    )
    val_loader: DataLoader = DataLoader(DeviceDataset(val_records, encoder), batch_size=batch_size)
    unseen_loader: DataLoader = DataLoader(DeviceDataset(unseen_records, encoder), batch_size=batch_size)

    vocab_sizes: dict[str, int] = {col: encoder.vocab_size(col) for col in CATEGORICAL_COLUMNS}
    model: TTFModel = TTFModel(
        vocab_sizes=vocab_sizes,
        embedding_dims=config["model"]["embedding_dims"],
        hidden_layers=config["model"]["hidden_layers"],
    )

    criterion: nn.Module = nn.MSELoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    unk_mask_prob: float = config["training"]["unk_mask_prob"]

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val: float = float("inf")
    best_epoch: int = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss: float = run_epoch(model, train_loader, criterion, optimizer, unk_mask_prob)
        val_loss: float = run_epoch(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch:03d} | treino: {train_loss:.2f} | validação: {val_loss:.2f}")

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)

    train_metrics: dict[str, float] = compute_metrics(model, train_loader, TOLERANCE_MONTHS)
    val_metrics: dict[str, float] = compute_metrics(model, val_loader, TOLERANCE_MONTHS)
    unseen_metrics: dict[str, float] = compute_metrics(model, unseen_loader, TOLERANCE_MONTHS)

    print(format_metrics("treino    ", train_metrics))
    print(format_metrics("validação ", val_metrics))
    print(format_metrics("modelo novo", unseen_metrics))
    print(f"melhor epoch: {best_epoch} | MSE validação: {best_val:.2f}")

    checkpoint_path: Path = bundle_dir / "ttf_model.pt"
    torch.save(best_state, checkpoint_path)
    digest: str = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest: dict[str, Any] = {
        "embedding_dims": config["model"]["embedding_dims"],
        "hidden_layers": config["model"]["hidden_layers"],
        "vocab_sizes": vocab_sizes,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "unseen_model_mae": unseen_metrics["mae"],
        "unseen_model_rmse": unseen_metrics["rmse"],
        "state_dict_sha256": digest,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    report: str = "\n".join(
        [
            format_metrics("treino     ", train_metrics),
            format_metrics("validação  ", val_metrics),
            format_metrics("modelo novo", unseen_metrics),
            f"melhor epoch: {best_epoch} | MSE validação: {best_val:.2f}",
        ]
    )
    (ROOT / "checkpoints" / "data.txt").write_text(report + "\n")

    plots: Path = ROOT / "checkpoints"
    plot_loss_curve(train_losses, val_losses, plots / "loss_curve.png")
    plot_prediction_error(model, val_loader, plots / "prediction_error.png")
    plot_error_by_range(model, val_loader, plots / "error_by_range.png")
    plot_embeddings(model, encoder, "manufacturer", plots / "embedding_manufacturer.png")
    plot_embeddings(model, encoder, "category", plots / "embedding_category.png")


if __name__ == "__main__":
    main()

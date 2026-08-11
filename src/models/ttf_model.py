import torch
import torch.nn as nn

from src.data.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


class TTFModel(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        hidden_layers: list[int],
    ) -> None:
        super().__init__()

        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(vocab_sizes[col], embedding_dims[col])
                for col in CATEGORICAL_COLUMNS
            }
        )

        input_dim = sum(embedding_dims[col] for col in CATEGORICAL_COLUMNS) + len(NUMERIC_COLUMNS)

        layers: list[nn.Module] = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embedded = [
            self.embeddings[col](categorical[:, i]) for i, col in enumerate(CATEGORICAL_COLUMNS)
        ]
        features = torch.cat(embedded + [numeric], dim=1)
        return self.mlp(features).squeeze(1)

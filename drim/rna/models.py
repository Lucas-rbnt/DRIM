import torch.nn as nn
import torch
from einops.layers.torch import Rearrange


class RNAEncoder(nn.Module):
    """
    A vanilla encoder based on 1-d convolution for RNA data.
    """

    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(8),
            nn.Dropout(dropout),
            nn.Conv1d(8, 32, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Conv1d(256, embedding_dim, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).squeeze(-1)
        return x


class RNAEncoderMlp(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(18000),
            nn.Linear(18000, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).squeeze(-1)


class RNADecoder(nn.Module):
    """
    A vanilla decoder based on 1-d transposed convolution for RNA data.
    """

    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 20),
            Rearrange("b (n e) -> b n e", e=20),
            nn.GELU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(embedding_dim, 256, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(256, 128, 9, 3, 2),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(128, 64, 9, 3, 1),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(64, 32, 9, 3, 4),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(32, 8, 9, 3, 1),
            nn.GELU(),
            nn.BatchNorm1d(8),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(8, 1, 9, 3, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x

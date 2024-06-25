import torch
import torch.nn as nn
import torchvision
from typing import Tuple, Union
from einops import repeat
from .func import Transformer
from einops.layers.torch import Rearrange


class ResNetWrapperSimCLR(nn.Module):
    """
    A wrapper for the ResNet34 model with a projection head for SimCLR.
    """

    def __init__(self, out_dim: int, projection_head: bool = True) -> None:
        super().__init__()
        self.encoder = torchvision.models.resnet34(pretrained=False)
        self.encoder.fc = nn.Identity()
        if projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, out_dim)
            )

    def forward(
        self, x: torch.tensor
    ) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        x = self.encoder(x)
        if hasattr(self, "projection_head"):
            return self.projection_head(x), x
        else:
            return x


class WSIEncoder(nn.Module):
    """
    A attention-based encoder for WSI data.
    """

    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        heads: int,
        dim: int = 512,
        pool: str = "cls",
        dim_head: int = 64,
        mlp_dim: int = 128,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = (
            nn.Identity() if embedding_dim == dim else nn.Linear(dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)

        return x


class WSIDecoder(nn.Module):
    """
    A vanilla mlp-based decoder for WSI data.
    """

    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(256, 5120),
            Rearrange("b (p e) -> b p e", p=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

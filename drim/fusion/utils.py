import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional


def softmax_one(x, dim=None):
    """
    Quiet softmax function as presented by Evan Miller in his blog post "Attention is Off by One".

    https://www.evanmiller.org/attention-is-off-by-one.html
    https://github.com/kyegomez/AttentionIsOFFByOne
    """
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)

    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Adapted from lucidrains vit-pytorch. It handles mask alongside the features vector.
    """

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask_value = torch.finfo(dots.dtype).min
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            dots.masked_fill_(~mask, mask_value)

        attn = softmax_one(dots, dim=-1)
        attn_d = self.dropout(attn)

        out = torch.matmul(attn_d, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn

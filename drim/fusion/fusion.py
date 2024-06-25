import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import torch
import random
from einops import repeat
from .utils import MultiHeadSelfAttention, FeedForward

__all__ = ["ShallowFusion", "TensorFusion", "MaskedAttentionFusion"]


class ShallowFusion(nn.Module):
    """
    Class computing the simplest fusion methods. Adapted from MultiSurv (https://www.nature.com/articles/s41598-021-92799-4?proof=t).
    It also contains the code from  Cheerla et al. (https://github.com/gevaertlab/MultimodalPrognosis/blob/master/experiments/chart4.py).
    """

    def __init__(self, fusion_type: str = "mean") -> None:
        super().__init__()
        self.fusion_type = fusion_type

    def forward(
        self, x: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor] = None
    ) -> torch.Tensor:
        if self.fusion_type == "mean":
            x = torch.stack(list(x.values()), dim=0).mean(dim=0)
        elif self.fusion_type == "max":
            x, _ = torch.stack(list(x.values()), dim=0).max(dim=0)
        elif self.fusion_type == "sum":
            x = torch.stack(list(x.values()), dim=0).sum(dim=0)
        elif self.fusion_type == "concat":
            x = torch.cat(list(x.values()), dim=-1)
        elif self.fusion_type == "masked_mean":
            x = self.masked_mean(x, mask)

        return x

    @staticmethod
    def masked_mean(x, mask):
        num = sum((x[modality] * mask[modality][:, None]) for modality in x.keys())
        den = sum(mask[modality] for modality in mask.keys())[:, None].float()
        return num / den


class TensorFusion(nn.Module):
    """
    Multimodal Tensor Fusion via Kronecker Product and Gating-Based Attention derived from [1, 2].

    [1] https://ieeexplore.ieee.org/document/9186053
    [2] https://link.springer.com/chapter/10.1007/978-3-030-87240-3_64

    NB: This is an extension of the original loss, which can take more than 3 modalities.

    This fusion is not particularly well adapted to missing modalities and its number of trainable parameters can go high.
    Be careful with the number of modalities and the size of the tensors. Moreover it is not invariant to modality permutation.

    For instance: [modalities (m), input dimension (i), projected dimension (p), output dimension (o) ~ number of parameters (millions)]
    m: 3, i: 128,  p: 128,  o: 128 ~ 280M
    m: 4, i: 16,  p: 16,  o: 16 ~ 1M4
    m: 4, i: 32,  p: 32,  o: 32 ~ 38M
    """

    def __init__(
        self,
        modalities: List[str],
        input_dim: int,
        projected_dim: int,
        output_dim: int,
        gate: bool = True,
        skip: bool = True,
        dropout: float = 0.1,
        pairs: Optional[Dict[str, str]] = None,
    ) -> None:
        # raise a warning because of the important number of trainable parameters.
        super().__init__()
        self.skip = skip
        self.modalities = sorted(modalities)
        self.dropout = dropout
        self.gate = gate
        if gate:
            if pairs:
                self.pairs = pairs
            else:
                self.pairs = {}
                for modality in self.modalities:
                    i = self.modalities.index(modality)
                    j = i
                    while j == i:
                        j = random.randint(0, len(self.modalities) - 1)
                    self.pairs[modality] = self.modalities[j]

        for modality in self.modalities:
            if gate:
                setattr(
                    self,
                    f"{modality.lower()}_linear",
                    nn.Sequential(nn.Linear(input_dim, projected_dim), nn.ReLU()),
                )
                setattr(
                    self,
                    f"{modality.lower()}_bilinear",
                    nn.Bilinear(input_dim, input_dim, projected_dim),
                )
                setattr(
                    self,
                    f"{modality.lower()}_last_linear",
                    nn.Sequential(
                        nn.Linear(projected_dim, projected_dim),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                    ),
                )
            else:
                setattr(
                    self,
                    f"{modality.lower()}_proj",
                    nn.Sequential(nn.Linear(input_dim, projected_dim), nn.ReLU()),
                )

        self.dropout = nn.Dropout(dropout)
        self.post_fusion = nn.Sequential(
            nn.Linear((input_dim + 1) ** len(modalities), output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        if self.skip:
            self.skip = nn.Sequential(
                nn.Linear(output_dim + ((input_dim + 1) * len(modalities)), output_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )

    def forward(self, x: Dict[str, torch.Tensor], mask=None) -> torch.Tensor:
        if self.gate:
            h = {k: getattr(self, f"{k.lower()}_linear")(v) for k, v in x.items()}
            x = {
                k: getattr(self, f"{k.lower()}_bilinear")(v, x[self.pairs[k]])
                for k, v in x.items()
            }
            x = {
                k: getattr(self, f"{k.lower()}_last_linear")(nn.Sigmoid()(x[k]) * h[k])
                for k, v in x.items()
            }

        else:
            x = {k: getattr(self, f"{k.lower()}_proj")(v) for k, v in x.items()}

        x = {
            k: torch.cat((v, torch.FloatTensor(v.shape[0], 1).fill_(1).to(v.device)), 1)
            for k, v in x.items()
        }
        out = torch.bmm(
            x[self.modalities[0]].unsqueeze(2), x[self.modalities[1]].unsqueeze(1)
        ).flatten(start_dim=1)
        for modality in self.modalities[2:]:
            out = torch.bmm(out.unsqueeze(2), x[modality].unsqueeze(1)).flatten(
                start_dim=1
            )

        out = self.dropout(out)
        out = self.post_fusion(out)
        if self.skip:
            out = torch.cat((out, *list(x.values())), 1)
            out = self.skip(out)

        return out


class MaskedAttentionFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        MultiHeadSelfAttention(
                            dim, heads=heads, dim_head=dim_head, dropout=dropout
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(
        self, x: Dict[str, torch.Tensor], mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        keys = list(x.keys())
        x = torch.stack([x[k] for k in keys], 1)
        scores = []
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0]).to(x.device)
        x = torch.cat((cls_tokens, x), dim=1)
        if mask is not None:
            mask = torch.stack(
                [torch.ones(x.shape[0]).bool().to(mask[keys[0]].device)]
                + [mask[k] for k in keys],
                1,
            )

        for attn, ff in self.layers:
            temp_x, score = attn(x, mask=mask)
            scores.append(score)
            x = temp_x + x
            x = ff(x) + x
        x = x[:, 0]
        scores = self._get_attn_scores(torch.stack(scores))
        return self.norm(x), {k: scores[:, i] for i, k in enumerate(keys)}

    @staticmethod
    def _get_attn_scores(x):
        return x.mean(dim=2)[:, :, 0, 1:].mean(0)

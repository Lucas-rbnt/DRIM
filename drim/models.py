from typing import Dict, Optional
import torch.nn as nn
import torch


class _BaseWrapper(nn.Module):
    """
    Base wrapper for taking the final embedding and linking to the survival task
    """

    def __init__(
        self, encoder: nn.Module, embedding_dim: int, n_outs: int, device: str = "cuda"
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.final = nn.Linear(embedding_dim, n_outs).to(device)
        self.device = device


class UnimodalWrapper(_BaseWrapper):
    """
    Survival wrapper taking an encoder et linking it to the survival task
    """

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """
        Args:
            x:
                tensor containing the raw input (bsz, n_features)
            return_embedding:
                boolean indicating if the pre-survival layer embedding will be used

        Returns
            Whether the predicted hazards or the predicted hazards and the last embedding.
        """
        x = x.to(self.device)
        x = self.encoder(x)
        if return_embedding:
            return self.final(x), x
        else:
            return self.final(x)


class MultimodalWrapper(_BaseWrapper):
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Dict associating each modality (key) to the raw input tensor (bsz, n_features)
            mask:
                Dict associating each modality (key) to a boolean tensor of size (bsz,) indicating if the modality is indeed present in the minibatch
            return_embedding:
                boolean indicating if the pre-survival layer embedding will be used

        Returns
            Whether the predicted hazards or the predicted hazards and the last embedding.
        """
        x = self.encoder(x, mask=mask, return_embedding=return_embedding)
        if not isinstance(x, tuple):
            # for fine-tuning or simple multimodal training
            return self.final(x.to(self.device))
        else:
            if len(x) == 2 and isinstance(x[0], dict):
                # for the self supervised settings
                return x
            else:
                out = self.final(x[0].to(self.device))
                if len(x) == 2:
                    # auxiliary training
                    return out, x[1]
                else:
                    # disentangled training
                    return out, x[1], x[2]


class Discriminator(nn.Module):
    """
    Generic discriminator for the adversarial training
    """

    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(1024, embedding_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

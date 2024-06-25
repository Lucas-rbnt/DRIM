from typing import Dict
import torch
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    """
    Implementation of the traditionnal discriminator loss to approximate Jensen-Shannon Divergence.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute discriminator loss

        Args:
            real_logits:
                torch.tensor containing logits extracted from p(x)p(y), size (bsz, 1)
            fake_logits:
                torch.tensor containing logits extracted from p(x, y), size (bsz, 1)

        Returns:
            The computed scalar loss.
        """
        # Discriminator should predict real logits as logits from the real distribution
        discriminator_real = torch.nn.functional.binary_cross_entropy_with_logits(
            input=real_logits, target=torch.ones_like(real_logits)
        )
        # Discriminator should predict fake logits as logits from the generated distribution
        discriminator_fake = torch.nn.functional.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.zeros_like(fake_logits)
        )

        discriminator_loss = discriminator_real + discriminator_fake

        return discriminator_loss


class ContrastiveLoss(nn.Module):
    """
    This code is adapted from the code for supervised contrastive learning (https://arxiv.org/pdf/2004.11362.pdf).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ) -> None:
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self, x: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute shared loss for our disentangled framework.

        Args:
            x:
                Dict associating each modality (key) to the corresponding embedding, tensor of size (bsz, d)
            mask:
                Dict associating each modality (key) to a boolean tensor of size (bsz,) indicating if the modality is indeed present in the minibatch
        Returns:
            The computed scalar loss.
        """
        keys = list(x.keys())
        features = torch.stack([x[key] for key in keys], 1)
        modality_mask = torch.stack([mask[key] for key in keys], 1)
        batch_size = features.shape[0]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        modality_mask = torch.cat(torch.unbind(modality_mask, dim=1), dim=0).view(1, -1)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = torch.eye(batch_size, dtype=torch.float32).to(modality_mask.device)
        mask = mask.repeat(contrast_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            (modality_mask[:, None, :, None] * modality_mask[:, None, None, :])
            .squeeze()
            .long(),
            1,
            torch.arange(batch_size * contrast_count)
            .view(-1, 1)
            .to(modality_mask.device),
            0,
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss


class MMOLoss(nn.Module):
    """
    Loss function used in the paper:
    Deep Orthogonal Fusion: Multimodal: Prognostic Biomarker Discovery Integrating Radiology,
    Pathology, Genomic, and Clinical Data.
    https://link.springer.com/chapter/10.1007/978-3-030-87240-3_64

    N.B: In the paper and by design, this loss is not well-suited to deal with missing modality, we propose here an alternative.
    """

    def __init__(self) -> None:
        super(MMOLoss, self).__init__()

    def forward(
        self, x: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute auxiliary orthogonal loss.

        Args:
            x:
                Dict associating each modality (key) to the corresponding embedding, tensor of size (bsz, d)
            mask:
                Dict associating each modality (key) to a boolean tensor of size (bsz,) indicating if the modality is indeed present in the minibatch,
        Returns:
            The computed scalar loss.
        """
        if mask is None:
            mask = {key: torch.ones_like(x[key]).bool() for key in x.keys()}

        loss = 0.0
        for key in x.keys():
            loss += torch.max(
                torch.tensor(1), torch.linalg.matrix_norm(x[key][mask[key]], ord="nuc")
            )

        full = torch.stack(list(x.values()))[torch.stack(list(mask.values()))]
        loss -= torch.linalg.matrix_norm(full, ord="nuc")
        loss /= full.size(0)
        return loss

from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from warnings import warn
import torch


class ContrastiveLoss(_Loss):
    """
    Deeply inspired and copy/paste from a previous PR on MONAI (https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/contrastive.py)

    Compute the Contrastive loss defined in:

        Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International
        conference on machine learning. PMLR, 2020. (http://proceedings.mlr.press/v119/chen20j.html)

    Adapted from:
        https://github.com/Sara-Ahmed/SiT/blob/1aacd6adcd39b71efc903d16b4e9095b97dda76f/losses.py#L5

    """

    def __init__(
        self, temperature: float = 0.5, k: int = 3, batch_size: int = -1
    ) -> None:
        """
        Args:
            temperature: Can be scaled between 0 and 1 for learning from negative samples, ideally set to 0.5.

        Raises:
            ValueError: When an input of dimension length > 2 is passed
            ValueError: When input and target are of different shapes

        """
        super().__init__()
        self.temperature = temperature
        self.k = k

        if batch_size != -1:
            warn(
                "batch_size is no longer required to be set. It will be estimated dynamically in the forward call"
            )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be B[F].
            target: the shape should be B[F].
        """
        if len(target.shape) > 2 or len(input.shape) > 2:
            raise ValueError(
                f"Either target or input has dimensions greater than 2 where target "
                f"shape is ({target.shape}) and input shape is ({input.shape})"
            )

        if target.shape != input.shape:
            raise ValueError(
                f"ground truth has differing shape ({target.shape}) from input ({input.shape})"
            )

        temperature_tensor = torch.as_tensor(self.temperature).to(input.device)
        batch_size = input.shape[0]

        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)
        negatives_mask = torch.clone(negatives_mask.type(torch.float)).to(input.device)

        repr = torch.cat([input, target], dim=0)
        sim_matrix = F.cosine_similarity(repr.unsqueeze(1), repr.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / temperature_tensor)
        denominator = negatives_mask * torch.exp(sim_matrix / temperature_tensor)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        # create pos mask
        self_mask = torch.eye(
            sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device
        )
        pos_mask = self_mask.roll(shifts=sim_matrix.shape[0] // 2, dims=0)
        sim_matrix.masked_fill_(self_mask, -9e15)
        comb_sim = torch.cat(
            [
                sim_matrix[pos_mask][:, None],  # First position positive example
                sim_matrix.masked_fill(pos_mask, -9e15),
            ],
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        return (
            torch.sum(loss_partial) / (2 * batch_size),
            (sim_argsort == 0).float().mean(),
            (sim_argsort < self.k).float().mean(),
            1 + sim_argsort.float().mean(),
        )

from ..datasets import _BaseDataset
from typing import Dict, List, Tuple, Union
import torch


class MultimodalDataset(_BaseDataset):
    def __init__(
        self, datasets: Dict[str, _BaseDataset], return_mask: bool = True
    ) -> None:
        modalities = list(datasets.keys())
        self._modality_sanity_check(modalities)
        if return_mask:
            for modality in modalities:
                assert datasets[
                    modality
                ].return_mask, f"The dataset for modality {modality} does not return a mask, please set return_mask to False"
        self.return_mask = return_mask
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets.values().__iter__().__next__())

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, bool]], Dict[str, torch.Tensor]
    ]:
        x = {}
        if self.return_mask:
            mask = {}

        for modality in self.datasets.keys():
            out = self.datasets[modality][idx]
            if self.return_mask:
                x[modality], mask[modality] = out[0], out[1]
            else:
                x[modality] = out

        if self.return_mask:
            return x, mask
        else:
            return x

    @staticmethod
    def _modality_sanity_check(
        modalities,
        available_modalities: List[str] = ["RNA", "MRI", "Clinical", "DNAm", "WSI"],
    ) -> None:
        for modality in modalities:
            assert (
                modality in available_modalities
            ), f"The requested modality: {modality} is not available, please choose modalities among {available_modalities}"

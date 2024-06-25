import torch
import pandas as pd
from typing import Union, Tuple
from ..datasets import _BaseDataset


class DNAmDataset(_BaseDataset):
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, bool], torch.Tensor]:
        sample = self.dataframe.iloc[idx]
        if not pd.isna(sample.DNAm):
            out = (
                torch.from_numpy(pd.read_csv(sample.DNAm).Beta_value.fillna(0.0).values)
                .float()
                .unsqueeze(0)
            )
            mask = True
        else:
            out = torch.zeros(1, 25978).float()
            mask = False

        if self.return_mask:
            return out, mask
        else:
            return out

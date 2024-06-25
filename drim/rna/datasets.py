from ..datasets import _BaseDataset
import pandas as pd
import torch
from typing import Union, Tuple
import numpy as np


def log_transform(x):
    return np.log(x + 1)


class RNADataset(_BaseDataset):
    """
    Simple dataset for RNA data.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        preprocessor: "sklearn.pipeline.Pipeline",
        return_mask: bool = False,
    ) -> None:
        super().__init__(dataframe, return_mask)
        self.preprocessor = preprocessor

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, bool], torch.Tensor]:
        sample = self.dataframe.iloc[idx]
        if not pd.isna(sample.RNA):
            out = torch.from_numpy(
                self.preprocessor.transform(
                    pd.read_csv(sample.RNA)["fpkm_uq_unstranded"].values.reshape(1, -1)
                )
            ).float()
            mask = True
        else:
            out = torch.zeros(1, 16304).float()
            mask = False

        if self.return_mask:
            return out, mask
        else:
            return out

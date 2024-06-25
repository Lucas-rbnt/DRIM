from abc import abstractmethod
import pandas as pd
import torch
from typing import Any, Tuple


__all__ = ["SurvivalDataset"]


class _BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, return_mask: bool = False) -> None:
        self.dataframe = dataframe
        self.return_mask = return_mask

    @abstractmethod
    def __getitem__(self, idx: int):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataframe)


class SurvivalDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset: torch.utils.data.Dataset, time: torch.tensor, event: torch.tensor
    ) -> None:
        self.dataset = dataset
        self.time, self.event = torch.from_numpy(time), torch.from_numpy(event)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        return self.dataset[idx], self.time[idx], self.event[idx]

    def __len__(self) -> int:
        return len(self.dataset)

import numpy as np
import pandas as pd
from typing import List, Union, Dict, Any, Tuple
import torch
import random


def seed_everything(seed: int) -> None:
    import monai

    monai.utils.set_determinism(seed=seed, additional_settings=None)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def prepare_data(
    dataframe: pd.DataFrame, modalities: Union[List[str], str], min_k_modality: int = 2
) -> pd.DataFrame:
    if isinstance(modalities, str):
        modalities = [modalities]

    # columns = ['project_id', 'submitter_id']
    # dataframe = dataframe.drop(columns=columns)

    if len(modalities) == 1:
        try:
            mask = ~dataframe[modalities[0]].isna()
            dataframe = dataframe.loc[mask]
        except:
            pass

    else:
        temp_df = dataframe[modalities]
        dataframe = dataframe[temp_df.count(axis=1) >= min_k_modality]

    return dataframe


get_target_survival = lambda df: (df["time"].values, df["event"].values)


def log_transform(x: np.ndarray) -> np.ndarray:
    return np.log(1 + x)


def create_nan_dataframe(
    num_row: int, num_col: int, name_col: List[str]
) -> pd.DataFrame:
    df = pd.DataFrame(np.zeros((num_row, num_col)), columns=name_col)
    df[:] = np.nan
    return df


def interpolate_dataframe(dataframe: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    dataframe.reset_index(inplace=True)
    dataframe_list = []
    for i, idx in enumerate(dataframe.index):
        df_temp = dataframe[dataframe.index == idx]
        dataframe_list.append(df_temp)
        if i != len(dataframe) - 1:
            dataframe_list.append(
                create_nan_dataframe(n, df_temp.shape[1], df_temp.columns)
            )

    dataframe = pd.concat(dataframe_list).interpolate("linear")
    dataframe = dataframe.set_index("index")
    return dataframe


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataframes(fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataframe = pd.read_csv("data/files/train_brain.csv")
    dataframe_train = (
        dataframe[dataframe["split"] != fold].copy().drop(columns=["split"])
    )
    dataframe_val = dataframe[dataframe["split"] == fold].copy().drop(columns=["split"])
    dataframe_test = pd.read_csv("data/files/test_brain.csv")
    return {"train": dataframe_train, "val": dataframe_val, "test": dataframe_test}

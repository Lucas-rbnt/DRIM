# Standard libraries
from typing import Dict

# Third-party libraries
import pandas as pd
from omegaconf import DictConfig
from pycox.models import LogisticHazard

# Local dependencies
from drim.utils import prepare_data, get_target_survival, log_transform


def get_encoder(modality: str, cfg: DictConfig):
    if modality == "DNAm":
        from drim.dnam import DNAmEncoder

        encoder = DNAmEncoder(
            embedding_dim=cfg.general.dim, dropout=cfg.general.dropout
        )
    elif modality == "RNA":
        from drim.rna import RNAEncoder

        encoder = RNAEncoder(embedding_dim=cfg.general.dim, dropout=cfg.general.dropout)
    elif modality == "WSI":
        from drim.wsi import WSIEncoder

        encoder = WSIEncoder(
            cfg.general.dim,
            depth=1,
            heads=8,
            dropout=cfg.general.dropout,
            emb_dropout=cfg.general.dropout,
        )
    elif modality == "MRI":
        from drim.mri import MRIEmbeddingEncoder

        encoder = MRIEmbeddingEncoder(
            embedding_dim=cfg.general.dim, dropout=cfg.general.dropout
        )
    else:
        raise NotImplementedError(f"Modality {modality} not implemented")

    return encoder


def get_decoder(modality: str, cfg: DictConfig):
    if modality == "DNAm":
        from drim.dnam import DNAmDecoder

        decoder = DNAmDecoder(
            embedding_dim=cfg.general.dim, dropout=cfg.general.dropout
        )
    elif modality == "RNA":
        from drim.rna import RNADecoder

        decoder = RNADecoder(embedding_dim=cfg.general.dim, dropout=cfg.general.dropout)
    elif modality == "WSI":
        from drim.wsi import WSIDecoder

        decoder = WSIDecoder(cfg.general.dim, dropout=cfg.general.dropout)
    elif modality == "MRI":
        from drim.mri import MRIEmbeddingDecoder

        decoder = MRIEmbeddingDecoder(
            embedding_dim=cfg.general.dim, dropout=cfg.general.dropout
        )
    else:
        raise NotImplementedError(f"Modality {modality} not implemented")

    return decoder


def get_datasets(
    dataframes: Dict[str, pd.DataFrame],
    modality: str,
    fold: int,
    return_mask: bool = False,
):
    if modality == "DNAm":
        from drim.dnam import DNAmDataset

        datasets = {
            split: DNAmDataset(dataframe, return_mask=return_mask)
            for split, dataframe in dataframes.items()
        }
    elif modality == "RNA":
        from drim.rna import RNADataset
        from joblib import load

        rna_processor = load(f"./data/rna_preprocessors/trf_{int(fold)}.joblib")
        datasets = {
            split: RNADataset(dataframe, rna_processor, return_mask=return_mask)
            for split, dataframe in dataframes.items()
        }
    elif modality == "WSI":
        from drim.wsi import WSIDataset

        datasets = {}
        datasets["train"] = WSIDataset(
            dataframes["train"], k=10, is_train=True, return_mask=return_mask
        )
        datasets["val"] = WSIDataset(
            dataframes["val"], k=10, is_train=False, return_mask=return_mask
        )
        datasets["test"] = WSIDataset(
            dataframes["test"], k=10, is_train=False, return_mask=return_mask
        )

    elif modality == "MRI":
        from drim.mri import MRIEmbeddingDataset

        datasets = {
            split: MRIEmbeddingDataset(dataframe, return_mask=return_mask)
            for split, dataframe in dataframes.items()
        }
    else:
        raise NotImplementedError(f"Modality {modality} not implemented")

    return datasets


def get_targets(dataframes: Dict[str, pd.DataFrame], n_outs: int):
    labtrans = LogisticHazard.label_transform(n_outs)
    labtrans.fit(*get_target_survival(dataframes["train"]))

    return {
        split: labtrans.transform(*get_target_survival(dataframe))
        for split, dataframe in dataframes.items()
    }, labtrans.cuts

# Standard libraries
from typing import Union, List
from collections import defaultdict

# Third-party libraries
import pandas as pd
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from pycox.evaluation import EvalSurv

# Local dependencies
from drim.helpers import get_datasets, get_targets, get_encoder
from drim.logger import logger
from drim.utils import (
    interpolate_dataframe,
    log_transform,
    seed_everything,
    get_dataframes,
)
from drim.multimodal import MultimodalDataset, DRIMSurv, MultimodalModel
from drim.datasets import SurvivalDataset
from drim.models import MultimodalWrapper


def prepare_data(
    dataframe: pd.DataFrame, modalities: Union[List[str], str]
) -> pd.DataFrame:
    available_modalities = ["DNAm", "WSI", "RNA", "MRI"]

    # columns = ['project_id', 'submitter_id']
    # dataframe = dataframe.drop(columns=columns)
    # get remaining modalities
    remaining = [
        modality for modality in available_modalities if modality not in modalities
    ]
    if len(modalities) == 1:
        try:
            mask = ~dataframe[modalities[0]].isna()
            dataframe = dataframe.loc[mask]
        except:
            pass

    else:
        temp_df = dataframe[modalities]
        dataframe = dataframe[temp_df.count(axis=1) >= len(modalities)]

    # put the whole columns of remaining modalities to NaN
    dataframe = dataframe.assign(**{modality: None for modality in remaining})
    return dataframe


@hydra.main(version_base=None, config_path="configs", config_name="robustness")
def main(cfg: DictConfig) -> None:
    logger.info("Starting multimodal robustness test.")
    logger.info("Method tested {}.", cfg.method)
    modality_combinations = [
        ["WSI", "MRI"],
        ["WSI", "RNA"],
        ["DNAm", "WSI"],
        ["DNAm", "RNA"],
        ["DNAm", "MRI"],
        ["RNA", "MRI"],
        ["DNAm", "WSI", "RNA"],
        ["DNAm", "WSI", "MRI"],
        ["DNAm", "RNA", "MRI"],
        ["WSI", "RNA", "MRI"],
        ["DNAm", "WSI", "RNA", "MRI"],
    ]
    for combination in modality_combinations:
        if cfg.method == "tensor":
            cfg.general.dim = 32
        logs = defaultdict(list)
        for fold in range(5):
            seed_everything(cfg.general.seed)
            dataframes = get_dataframes(fold)
            dataframes = {
                split: prepare_data(dataframe, combination)
                for split, dataframe in dataframes.items()
            }
            test_datasets = {}
            encoders = {}
            if cfg.method == "drim":
                encoders_u = {}

            for modality in ["DNAm", "WSI", "RNA", "MRI"]:
                datasets = get_datasets(dataframes, modality, fold, return_mask=True)
                test_datasets[modality] = datasets["test"]
                encoder = get_encoder(modality, cfg).cuda()
                encoders[modality] = encoder
                if cfg.method == "drim":
                    encoder_u = get_encoder(modality, cfg).cuda()
                    encoders_u[modality] = encoder_u

            targets, cut = get_targets(dataframes, cfg.general.n_outs)
            dataset_test = MultimodalDataset(test_datasets, return_mask=True)
            test_data = SurvivalDataset(dataset_test, *targets["test"])
            loader = torch.utils.data.DataLoader(
                test_data, shuffle=False, batch_size=24
            )
            if cfg.method == "drim":
                from drim.fusion import MaskedAttentionFusion

                fusion = MaskedAttentionFusion(
                    dim=cfg.general.dim, depth=1, heads=16, dim_head=64, mlp_dim=128
                )
                fusion_u = MaskedAttentionFusion(
                    dim=cfg.general.dim, depth=1, heads=16, dim_head=64, mlp_dim=128
                )
                fusion.cuda()
                fusion_u.cuda()
                encoder = DRIMSurv(
                    encoders_sh=encoders,
                    encoders_u=encoders_u,
                    fusion_s=fusion,
                    fusion_u=fusion_u,
                )
                model = MultimodalWrapper(
                    encoder, embedding_dim=cfg.general.dim, n_outs=cfg.general.n_outs
                )
                model.load_state_dict(
                    torch.load(f"./models/drimsurv_split_{int(fold)}.pth")
                )
            else:
                if cfg.method == "max":
                    from drim.fusion import ShallowFusion

                    fusion = ShallowFusion("max")
                elif cfg.method == "tensor":
                    from drim.fusion import TensorFusion

                    fusion = TensorFusion(
                        modalities=["DNAm", "WSI", "RNA", "MRI"],
                        input_dim=cfg.general.dim,
                        projected_dim=cfg.general.dim,
                        output_dim=cfg.general.dim,
                        dropout=0.0,
                    )
                elif cfg.method == "concat":
                    from drim.fusion import ShallowFusion

                    fusion = ShallowFusion("concat")

                fusion.cuda()
                encoder = MultimodalModel(encoders, fusion=fusion)
                if cfg.method == "concat":
                    size = cfg.general.dim * 4
                else:
                    size = cfg.general.dim
                model = MultimodalWrapper(
                    encoder, embedding_dim=size, n_outs=cfg.general.n_outs
                )
                if cfg.method == "max":
                    prefix = "vanilla"
                else:
                    prefix = "aux_contrastive"
                model.load_state_dict(
                    torch.load(f"./models/{prefix}_{cfg.method}_split_{int(fold)}.pth")
                )

            model.cuda()
            model.eval()
            hazards = []
            times = []
            events = []
            with torch.no_grad():
                for batch in loader:
                    data, time, event = batch
                    data, mask = data
                    outputs = model(data, mask, return_embedding=False)
                    hazards.append(outputs)
                    times.append(time)
                    events.append(event)

            times = torch.cat(times, dim=0).cpu().numpy()
            events = torch.cat(events, dim=0).cpu().numpy()
            hazards = interpolate_dataframe(
                pd.DataFrame(
                    (1 - torch.cat(hazards, dim=0).sigmoid())
                    .add(1e-7)
                    .log()
                    .cumsum(1)
                    .exp()
                    .cpu()
                    .numpy()
                    .transpose(),
                    cut,
                )
            )
            ev = EvalSurv(hazards, times, events, censor_surv="km")
            c_index = ev.concordance_td()
            ibs = ev.integrated_brier_score(np.linspace(0, times.max()))
            CS = (c_index + (1 - ibs)) / 2
            logs["c_index"].append(c_index)
            logs["ibs"].append(ibs)
            logs["CS"].append(CS)

        logger.info(
            f"{combination} - CS: {np.mean(logs['CS']):.3f} $\pm$ {np.std(logs['CS']):.3f}"
        )


if __name__ == "__main__":
    main()

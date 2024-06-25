# Standard libraries
from collections import defaultdict

# Third-party libraries
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch
from pycox.models.loss import NLLLogistiHazardLoss
import numpy as np

# Local dependencies
from drim.models import UnimodalWrapper
from drim.trainers import BaseSurvivalTrainer
from drim.datasets import SurvivalDataset
from drim.logger import logger
from drim.utils import (
    seed_everything,
    seed_worker,
    prepare_data,
    get_dataframes,
    log_transform,
)
from drim.helpers import get_encoder, get_datasets, get_targets


@hydra.main(version_base=None, config_path="configs", config_name="unimodal")
def main(cfg: DictConfig) -> None:
    cv_metrics = defaultdict(list)
    # check if wandb key is in cfg
    if "wandb" in cfg:
        import wandb

        wandb_logging = True
        wandb.init(
            name=cfg.general.modalities,
            config={
                k: v for k, v in OmegaConf.to_container(cfg).items() if k != "wandb"
            },
            **cfg.wandb,
        )
    else:
        wandb_logging = False

    logger.info("Starting unimodal cross-validation.")
    logger.info("Modality used: {}.", cfg.general.modalities)
    for fold in range(cfg.general.n_folds):
        logger.info("Starting fold {}", fold)
        seed_everything(cfg.general.seed)
        # Load the data
        dataframes = get_dataframes(fold)
        # take only the intersection between multimodal data and unimodal to ensure fair comparisons
        dataframes_multi = {
            split: prepare_data(dataframe, ["DNAm", "WSI", "RNA", "MRI"])
            for split, dataframe in dataframes.items()
        }

        dataframes = {
            split: prepare_data(dataframe, cfg.general.modalities)
            for split, dataframe in dataframes.items()
        }

        dataframes = {
            split: dataframe[
                dataframe["submitter_id"].isin(dataframes_multi[split]["submitter_id"])
            ]
            for split, dataframe in dataframes.items()
        }
        cfg.general.save_path = (
            f"./models/{cfg.general.modalities}_split_{int(fold)}.pth"
        )
        for split, dataframe in dataframes.items():
            logger.info(f"{split} samples: {len(dataframe)}")

        # Load the model
        logger.info("Loading model and preparing corresponding dataset...")
        encoder = get_encoder(cfg.general.modalities, cfg)
        datasets = get_datasets(
            dataframes, cfg.general.modalities, fold, return_mask=False
        )
        targets, cuts = get_targets(dataframes, cfg.general.n_outs)
        train_data = SurvivalDataset(datasets["train"], *targets["train"])
        val_data = SurvivalDataset(datasets["val"], *targets["val"])
        test_data = SurvivalDataset(datasets["test"], *targets["test"])

        dataloaders = {
            "train": DataLoader(
                train_data, shuffle=True, worker_init_fn=seed_worker, **cfg.dataloader
            ),
            "val": DataLoader(
                val_data, shuffle=False, worker_init_fn=seed_worker, **cfg.dataloader
            ),
            "test": DataLoader(
                test_data, shuffle=False, worker_init_fn=seed_worker, **cfg.dataloader
            ),
        }

        model = UnimodalWrapper(encoder, cfg.general.dim, n_outs=cfg.general.n_outs)
        model = model.cuda()
        logger.info("Done!")

        optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer.params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **cfg.scheduler
        )

        task_criterion = NLLLogistiHazardLoss()
        trainer = BaseSurvivalTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            task_criterion=task_criterion,
            cfg=cfg,
            wandb_logging=wandb_logging,
            cuts=cuts,
        )

        trainer.fit()
        val_logs = trainer.evaluate("val")
        test_logs = trainer.evaluate("test")
        # add to cv_metrics
        for key, value in val_logs.items():
            cv_metrics[key].append(value)

        for key, value in test_logs.items():
            cv_metrics[key].append(value)

        logger.info("Fold {} done!", fold)

    # log first the mean ± std of the validation metrics
    logs = {}
    for key, value in cv_metrics.items():
        if key in [
            "test/c_index",
            "test/cs_score",
            "test/inbll",
            "test/ibs",
            "val/c_index",
            "val/cs_score",
            "val/inbll",
            "val/ibs",
        ]:
            mean, std = np.mean(value), np.std(value)
            logger.info(f"{key}: {mean:.4f} ± {std:.4f}")
            logs[f"fin/{'_'.join(key.split('/'))}_mean"] = mean
            logs[f"fin/{'_'.join(key.split('/'))}_std"] = std

    if wandb_logging:
        wandb.log(logs)
        wandb.finish()


if __name__ == "__main__":
    main()

# Standard libraries
from collections import defaultdict

# Third-party libraries
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.utils.data import DataLoader
import torch
from pycox.models.loss import NLLLogistiHazardLoss
import numpy as np

# Local dependencies
from drim.trainers import DRIMSurvTrainer
from drim.multimodal import MultimodalDataset, DRIMSurv
from drim.models import MultimodalWrapper, Discriminator
from drim.losses import ContrastiveLoss
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


@hydra.main(version_base=None, config_path="configs", config_name="multimodal")
def main(cfg: DictConfig) -> None:
    cv_metrics = defaultdict(list)
    # check if wandb key is in cfg
    if "wandb" in cfg:
        import wandb

        wandb_logging = True
        wandb.init(
            name="DRIMSurv_" + "_".join(cfg.general.modalities),
            config={
                k: v for k, v in OmegaConf.to_container(cfg).items() if k != "wandb"
            },
            **cfg.wandb,
        )
    else:
        wandb_logging = False
    logger.info("Starting multimodal cross-validation.")
    logger.info("Modalities used: {}.", cfg.general.modalities)
    for fold in range(cfg.general.n_folds):
        logger.info("Starting fold {}", fold)
        seed_everything(cfg.general.seed)
        # Load the data
        dataframes = get_dataframes(fold)
        dataframes = {
            split: prepare_data(dataframe, cfg.general.modalities)
            for split, dataframe in dataframes.items()
        }
        cfg.general.save_path = f"./models/drimsurv_split_{int(fold)}.pth"

        for split, dataframe in dataframes.items():
            logger.info(f"{split} samples: {len(dataframe)}")

        train_datasets = {}
        val_datasets = {}
        test_datasets = {}
        encoders = {}
        encoders_u = {}
        logger.info("Loading models and preparing corresponding dataset...")
        for modality in cfg.general.modalities:
            encoder = get_encoder(modality, cfg).cuda()
            encoders[modality] = encoder
            encoder_u = get_encoder(modality, cfg).cuda()
            encoders_u[modality] = encoder_u
            datasets = get_datasets(dataframes, modality, fold, return_mask=True)
            train_datasets[modality] = datasets["train"]
            val_datasets[modality] = datasets["val"]
            test_datasets[modality] = datasets["test"]

        targets, cuts = get_targets(dataframes, cfg.general.n_outs)

        dataset_train = MultimodalDataset(train_datasets, return_mask=True)
        dataset_val = MultimodalDataset(val_datasets, return_mask=True)
        dataset_test = MultimodalDataset(test_datasets, return_mask=True)
        train_data = SurvivalDataset(dataset_train, *targets["train"])
        val_data = SurvivalDataset(dataset_val, *targets["val"])
        test_data = SurvivalDataset(dataset_test, *targets["test"])

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

        if cfg.fusion.name in ["mean", "concat", "max", "sum", "masked_mean"]:
            from drim.fusion import ShallowFusion

            fusion = ShallowFusion(cfg.fusion.name)
            fusion_u = ShallowFusion(cfg.fusion.name)
        elif cfg.fusion.name == "maf":
            from drim.fusion import MaskedAttentionFusion

            fusion = MaskedAttentionFusion(
                dim=cfg.general.dim, dropout=cfg.general.dropout, **cfg.fusion.params
            )
            fusion_u = MaskedAttentionFusion(
                dim=cfg.general.dim, dropout=cfg.general.dropout, **cfg.fusion.params
            )
        elif cfg.fusion.name == "tensor":
            from drim.fusion import TensorFusion

            fusion = TensorFusion(
                modalities=cfg.general.modalities,
                input_dim=cfg.general.dim,
                projected_dim=cfg.general.dim,
                output_dim=cfg.general.dim,
                dropout=cfg.general.dropout,
            )
            fusion_u = TensorFusion(
                modalities=cfg.general.modalities + ["shared"],
                input_dim=cfg.general.dim,
                projected_dim=cfg.general.dim,
                output_dim=cfg.general.dim,
                dropout=cfg.general.dropout,
            )
        else:
            raise NotImplementedError

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
        logger.info("Done!")

        logger.info("Preparing discriminators..")
        discriminators = {
            k: Discriminator(
                embedding_dim=cfg.general.dim * 2, dropout=cfg.general.dropout
            ).to("cuda")
            for k in encoders.keys()
        }

        # define optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer.params)
        optimizers_dsm = {
            k: torch.optim.AdamW(
                discriminators[k].parameters(),
                lr=cfg.disentangled.dsm_lr,
                weight_decay=cfg.disentangled.dsm_wd,
            )
            for k in encoders.keys()
        }
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **cfg.scheduler
        )

        # define task criterion
        task_criterion = NLLLogistiHazardLoss()

        # define auxiliary criterion
        aux_loss = ContrastiveLoss()

        trainer = DRIMSurvTrainer(
            model=model,
            discriminators=discriminators,
            optimizer=optimizer,
            optimizers_dsm=optimizers_dsm,
            scheduler=scheduler,
            dataloaders=dataloaders,
            task_criterion=task_criterion,
            aux_loss=aux_loss,
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

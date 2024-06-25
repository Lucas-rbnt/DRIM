# Standard libraries
import argparse
import os

# Third-party libraries
import torch
import wandb
import numpy as np
from imutils import paths

# Local dependencies
from drim.utils import seed_everything, seed_worker
from drim.logger import logger
from drim.wsi.transforms import contrastive_wsi_transforms
from drim.wsi.datasets import PatchDataset
from drim.wsi.models import ResNetWrapperSimCLR
from drim.commons.optim_contrastive import training_loop_contrastive
from drim.commons.losses import ContrastiveLoss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        nargs="+",
        default=["/media/hddext/medical/TCGA-GBM_WSI", "/archive/medical/tcga_lgg/wsi"],
    )
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--project", type=str, default="DRIM")
    parser.add_argument("--n_cpus", type=int, default=30)
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1999)
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)
    # get every filepaths
    filepaths = []
    for path in args.data_path:
        temp_filepaths = list(paths.list_images(path))
        temp_filepaths = [
            filepath for filepath in temp_filepaths if "patches" in filepath
        ]
        filepaths += temp_filepaths

    filepaths = np.array(filepaths)
    # split patient into train and val by taking random 75% of patients for training

    train_idx = np.random.choice(
        np.arange(len(filepaths)), int(len(filepaths) * 0.75), replace=False
    )
    train_mask = np.zeros(len(filepaths), dtype=bool)
    train_mask[train_idx] = True
    train_filepaths = filepaths[train_idx]
    val_filepaths = filepaths[~train_mask]

    logger.info("Performing contrastive training on TCGA dataset.")
    logger.info("Number of training images : {}.", len(train_filepaths))
    logger.info("Number of validation images : {}.", len(val_filepaths))

    train_dataset = PatchDataset(train_filepaths, transforms=contrastive_wsi_transforms)
    val_dataset = PatchDataset(val_filepaths, transforms=contrastive_wsi_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    model = ResNetWrapperSimCLR(out_dim=args.out_dim, projection_head=True)

    logger.info(
        "Number of parameters : {}",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.n_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpus)))

    model.to(device)
    logger.info("Using {} gpus to train the model", args.n_gpus)

    contrastive_loss = ContrastiveLoss(temperature=args.temperature, k=args.k)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=5e-6
    )

    path_to_save = f"data/models/wsi_encoder.pth"
    wandb_logging = True if args.entity is not None else False
    if wandb_logging:
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"Pretraining_WSI",
            reinit=True,
            config=vars(args),
        )
    logger.info("Training started!")
    _, _ = training_loop_contrastive(
        model,
        args.epochs,
        contrastive_loss,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        device,
        path_to_save,
        wandb_logging=wandb_logging,
        k=args.k,
    )
    if wandb_logging:
        run.finish()
    logger.info("Training finished!")

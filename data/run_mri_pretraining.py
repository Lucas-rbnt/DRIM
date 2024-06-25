# Standard libraries
import argparse
import os

# Third-party libraries
import torch
import wandb
import numpy as np
import pandas as pd

# Local libraries
from drim.utils import seed_everything, seed_worker
from drim.logger import logger
from drim.mri.datasets import DatasetBraTSTumorCentered
from drim.mri.models import MRIEncoder
from drim.mri.transforms import get_tumor_transforms
from drim.commons.optim_contrastive import training_loop_contrastive
from drim.commons.losses import ContrastiveLoss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../TCGA/GBMLGG/MRI")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--modalities", nargs="+", default=["t1ce", "flair"])
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--project", type=str, default="DRIM")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--tumor_centered", type=bool, default=True)
    parser.add_argument("--n_cpus", type=int, default=40)
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1999)
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)
    patients = os.listdir(args.data_path)
    dataframe = pd.read_csv("data/files/dataframe_brain.csv")
    dataframe_test = dataframe[dataframe["group"] == "test"]
    # get patient ids where MRI is not NaN
    dataframe_test = dataframe_test[~dataframe_test["MRI"].isna()]
    patients_to_exclude = [
        patient_path.split("/")[-1] for patient_path in dataframe_test.MRI.values
    ]
    patients = [patient for patient in patients if patient not in patients_to_exclude]

    # split patient into train and val by taking random 70% of patients for training
    train_patients = np.random.choice(
        patients, int(len(patients) * 0.75), replace=False
    )
    val_patients = [patient for patient in patients if patient not in train_patients]
    logger.info("Performing contrastive training on BraTS dataset.")
    logger.info("Modalities used : {}.", args.modalities)
    if bool(args.tumor_centered):
        logger.info("Using tumor centered dataset.")
        category = "tumor"
        sizes = (64, 64, 64)
        train_dataset = DatasetBraTSTumorCentered(
            args.data_path,
            args.modalities,
            patients=train_patients,
            sizes=sizes,
            return_mask=False,
            transform=get_tumor_transforms(sizes),
        )
        val_dataset = DatasetBraTSTumorCentered(
            args.data_path,
            args.modalities,
            patients=val_patients,
            sizes=sizes,
            return_mask=False,
            transform=get_tumor_transforms(sizes),
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.n_cpus,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpus,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )

    model = MRIEncoder(projection_head=True, in_channels=len(args.modalities))

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
        optimizer, T_max=args.epochs, eta_min=5e-6
    )

    path_to_save = (
        f"data/models/{'-'.join(args.modalities)}_tumor{str(args.tumor_centered)}.pth"
    )
    # if a wandb entity is provided, log the training on wandb
    wandb_logging = True if args.entity is not None else False
    if wandb_logging:
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"Pretraining_MRI",
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

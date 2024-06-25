import torch.nn as nn
from typing import Dict, Optional
from torchinfo import summary
from collections import defaultdict
import numpy as np
from pycox.evaluation import EvalSurv
import wandb
from .utils import interpolate_dataframe
import torch
from omegaconf import DictConfig, OmegaConf
from .logger import logger
import pandas as pd
import tqdm
from .losses import DiscriminatorLoss


class _BaseTrainer:
    """
    Base trainer for all experiments
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        task_criterion: nn.modules.loss._Loss,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        cfg: DictConfig,
        wandb_logging: bool,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task_criterion = task_criterion
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.wandb_logging = wandb_logging

    def _print_summary(self) -> None:
        """
        Print the model summary and the number of trainable parameters
        """
        logger.info(
            "Trainable parameters {}",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
        logger.info(self.model)

    def _initialize_logger(self) -> None:
        """
        Initialize the WandB logger
        """
        if "wandb" in self.cfg:
            self.wandb_logging = True
            if not isinstance(self.cfg.general.modalities, str):
                name = "_".join(self.cfg.general.modalities)
            else:
                name = self.cfg.general.modalities
            wandb.init(
                name=self.prefix + name,
                config={
                    k: v
                    for k, v in OmegaConf.to_container(self.cfg).items()
                    if k != "wandb"
                },
                **self.cfg.wandb,
            )
        else:
            self.wandb_logging = False

    def put_model_on_correct_mode(self, split: str) -> None:
        if split == "train":
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

    def clear(self) -> None:
        if self.wandb_logging:
            wandb.finish()

        return

    def fit(self, epochs: Optional[int] = None) -> None:
        """
        Train the self.model for a given number of epochs
        """
        # self._initialize_logger()
        # self.wandb_logging = False
        self._print_summary()
        metrics = defaultdict(list)
        if not epochs:
            epochs = self.cfg.general.epochs
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info("-" * 10)
            metrics["lr"].append(self.optimizer.state_dict()["param_groups"][0]["lr"])

            train_losses = self.shared_loop(split="train")
            for key, value in train_losses.items():
                logger.info(f"Train {key} {value:.4f}")
                metrics[f"train/{key}"].append(value)

            val_metrics = self.shared_loop(split="val")
            for key, value in val_metrics.items():
                logger.info(f"Val {key} {value:.4f}")
                metrics[f"val/{key}"].append(value)

            if self.wandb_logging:
                wandb.log({key: value[-1] for key, value in metrics.items()})

            self.scheduler.step()
        torch.save(self.model.state_dict(), self.cfg.general.save_path)

    def evaluate(self, split: str) -> Dict[str, float]:
        outputs = self.shared_loop(split)
        for key, value in outputs.items():
            logger.info(f"{split} {key} {value:.4f}")
        if self.wandb_logging:
            # update outputs
            outputs = {f"{split}/{key}": value for key, value in outputs.items()}
            wandb.log(outputs)

        return outputs


class BaseSurvivalTrainer(_BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        task_criterion: nn.modules.loss._Loss,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        cfg: DictConfig,
        wandb_logging: bool,
        cuts: np.ndarray,
    ):
        super().__init__(
            model, optimizer, scheduler, task_criterion, dataloaders, cfg, wandb_logging
        )
        self.cuts = cuts

    def compute_survival_metrics(self, outputs, time, event):
        """
        Compute the survival metrics with the PyCox package.
        """
        hazard = torch.cat(outputs, dim=0)
        survival = (1 - hazard.sigmoid()).add(1e-7).log().cumsum(1).exp().cpu().numpy()
        survival = interpolate_dataframe(pd.DataFrame(survival.transpose(), self.cuts))
        evaluator = EvalSurv(
            survival, time.cpu().numpy(), event.cpu().numpy(), censor_surv="km"
        )
        c_index = evaluator.concordance_td()
        ibs = evaluator.integrated_brier_score(np.linspace(0, time.cpu().numpy().max()))
        inbll = evaluator.integrated_nbll(np.linspace(0, time.cpu().numpy().max()))
        cs_score = (c_index + (1 - ibs)) / 2
        return {"c_index": c_index, "ibs": ibs, "inbll": inbll, "cs_score": cs_score}

    def shared_loop(self, split: str) -> Dict[str, float]:
        self.put_model_on_correct_mode(split)
        total_task_loss = 0.0
        raw_predictions = []
        times = []
        events = []
        for batch in tqdm.tqdm(self.dataloaders[split]):
            data, time, event = batch
            # check if data contains mask
            if isinstance(data, list):
                data, mask = data
                outputs = self.model(data, mask, return_embedding=False)
            else:
                outputs = self.model(data, return_embedding=False)

            loss = self.task_criterion(
                outputs, time.to(outputs.device), event.to(outputs.device)
            )
            if split == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_task_loss += loss.item() * time.size(0)
            raw_predictions.append(outputs)
            times.append(time)
            events.append(event)

        outputs = {"task_loss": total_task_loss / len(self.dataloaders[split].dataset)}
        if split != "train":
            task_metrics = self.compute_survival_metrics(
                raw_predictions, torch.cat(times, dim=0), torch.cat(events, dim=0)
            )
            outputs.update(task_metrics)

        return outputs


class AuxSurvivalTrainer(BaseSurvivalTrainer):
    """
    This class assumes that an auxiliary loss is added to the survival loss. This auxiliary loss can be either MMO Loss or CL loss.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        task_criterion: nn.modules.loss._Loss,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        cfg: DictConfig,
        wandb_logging: bool,
        cuts: np.ndarray,
        aux_loss: torch.nn.modules.loss._Loss,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            scheduler,
            task_criterion,
            dataloaders,
            cfg,
            wandb_logging,
            cuts,
        )
        self.aux_loss = aux_loss

    def shared_loop(self, split: str) -> Dict[str, float]:
        self.put_model_on_correct_mode(split)
        raw_predictions = []
        embeddings = defaultdict(list)
        masks = defaultdict(list)
        total_task_loss = 0.0
        total_aux_loss = 0.0
        times = []
        events = []
        for batch in tqdm.tqdm(self.dataloaders[split]):
            data, time, event = batch
            # check if data contains mask
            if isinstance(data, list):
                data, mask = data
                outputs, batch_embeddings = self.model(
                    data, mask, return_embedding=True
                )
            else:
                outputs, batch_embeddings = self.model(data, return_embedding=True)

            task_loss = self.task_criterion(
                outputs, time.to(outputs.device), event.to(outputs.device)
            )
            # put mask on the same device as embeddings
            mask = {k: v.to(batch_embeddings[k].device) for k, v in mask.items()}
            aux_loss = self.aux_loss(batch_embeddings, mask)
            loss = task_loss + self.cfg.aux_loss.alpha * aux_loss
            if split == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_task_loss += task_loss.item() * time.size(0)
            total_aux_loss += aux_loss.item() * time.size(0)
            raw_predictions.append(outputs)
            times.append(time)
            events.append(event)

        outputs = {
            "task_loss": total_task_loss / len(self.dataloaders[split].dataset),
            "aux_loss": total_aux_loss / len(self.dataloaders[split].dataset),
        }
        if split != "train":
            task_metrics = self.compute_survival_metrics(
                raw_predictions, torch.cat(times, dim=0), torch.cat(events, dim=0)
            )
            embeddings = {k: torch.cat(v) for k, v in embeddings.items()}
            masks = {k: torch.cat(v) for k, v in masks.items()}
            outputs.update(task_metrics)

        return outputs


class DRIMSurvTrainer(AuxSurvivalTrainer):
    def __init__(
        self,
        model: nn.Module,
        discriminators: Dict[str, nn.Module],
        optimizer: torch.optim.Optimizer,
        optimizers_dsm: Dict[str, torch.optim.Optimizer],
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        task_criterion: nn.modules.loss._Loss,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        cfg: DictConfig,
        wandb_logging: bool,
        cuts: np.ndarray,
        aux_loss: torch.nn.modules.loss._Loss,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            scheduler,
            task_criterion,
            dataloaders,
            cfg,
            wandb_logging,
            cuts,
            aux_loss,
        )
        self.discriminators = discriminators
        self.optimizers_dsm = optimizers_dsm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator_loss = DiscriminatorLoss()

    def shared_loop(self, split: str) -> Dict[str, float]:
        self.put_model_on_correct_mode(split)
        raw_predictions = []
        embeddings = defaultdict(list)
        masks = defaultdict(list)
        total_task_loss = 0.0
        total_sh_loss = 0.0
        total_u_loss = 0.0
        total_discrimators_loss = {k: 0.0 for k in self.optimizers_dsm.keys()}
        times = []
        events = []
        for batch in tqdm.tqdm(self.dataloaders[split]):
            data, time, event = batch
            # check if data contains mask
            if isinstance(data, list):
                data, mask = data
                outputs, batch_embeddings_sh, batch_embeddings_u = self.model(
                    data, mask, return_embedding=True
                )
            else:
                outputs, batch_embeddings_sh, batch_embeddings_u = self.model(
                    data, return_embedding=True
                )

            # link to survival loss
            task_loss = self.task_criterion(
                outputs, time.to(outputs.device), event.to(outputs.device)
            )
            # compute shared loss
            sh_loss = self.aux_loss(
                {k: v.to(self.device) for k, v in batch_embeddings_sh.items()},
                {k: v.to(self.device) for k, v in mask.items()},
            )
            # put mask on cpu
            mask = {k: v.cpu() for k, v in mask.items()}
            # prepare inputs for unique encoders, x_r is suppose to be drawn from P(x, y) while x_r_prime is drawn from P(x)P(y)
            x_r = {
                k: torch.cat(
                    [
                        batch_embeddings_sh[k][mask[k]]
                        .detach()
                        .to(batch_embeddings_u[k].device),
                        batch_embeddings_u[k][mask[k]],
                    ],
                    dim=1,
                )
                for k in batch_embeddings_u.keys()
            }
            x_r_prime = {
                k: torch.cat(
                    [
                        torch.roll(
                            batch_embeddings_sh[k][mask[k]]
                            .detach()
                            .to(batch_embeddings_u[k].device),
                            1,
                            0,
                        ),
                        batch_embeddings_u[k][mask[k]],
                    ],
                    dim=1,
                )
                for k in batch_embeddings_sh.keys()
            }
            # compute unique loss
            u_loss = torch.tensor(0.0).to(self.device)
            for modality in x_r.keys():
                # if mask is empty then continue
                if sum(mask[modality]) <= 1:
                    continue
                # get fake logits
                fake_logits = self.discriminators[modality](x_r[modality])
                u_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                    input=fake_logits, target=torch.ones_like(fake_logits)
                ).to(self.device)

            # update shared and unique encoders according to their respective loss and the task loss
            loss_encoders = (
                task_loss
                + self.cfg.aux_loss.alpha * sh_loss
                + self.cfg.disentangled.gamma * u_loss
            )
            if split == "train":
                self.optimizer.zero_grad()
                loss_encoders.backward()
                self.optimizer.step()

            # update discriminators
            for modality in x_r.keys():
                # if mask is empty then continue
                if sum(mask[modality]) <= 1:
                    continue

                fake_logits = self.discriminators[modality](x_r[modality].detach())
                real_logits = self.discriminators[modality](
                    x_r_prime[modality].detach()
                )
                loss_dsm = self.discriminator_loss(
                    real_logits=real_logits, fake_logits=fake_logits
                )

                if split == "train":
                    self.optimizers_dsm[modality].zero_grad()
                    loss_dsm.backward()
                    self.optimizers_dsm[modality].step()

                total_discrimators_loss[modality] += loss_dsm.item() * time.size(0)

            total_task_loss += task_loss.item() * time.size(0)
            total_sh_loss += sh_loss.item() * time.size(0)
            total_u_loss += u_loss.item() * time.size(0)
            raw_predictions.append(outputs)
            times.append(time)
            events.append(event)

        outputs = {
            "shared_loss": total_sh_loss / len(self.dataloaders[split].dataset),
            "unique_loss": total_u_loss / len(self.dataloaders[split].dataset),
            "task_loss": total_task_loss / len(self.dataloaders[split].dataset),
        }
        outputs.update(
            {
                f"discriminator_{k}": v / len(self.dataloaders[split].dataset)
                for k, v in total_discrimators_loss.items()
            }
        )

        if split != "train":
            task_metrics = self.compute_survival_metrics(
                raw_predictions, torch.cat(times, dim=0), torch.cat(events, dim=0)
            )
            embeddings = {k: torch.cat(v) for k, v in embeddings.items()}
            masks = {k: torch.cat(v) for k, v in masks.items()}
            outputs = {**outputs, **task_metrics}

        return outputs


class DRIMUTrainer(DRIMSurvTrainer):
    def __init__(
        self,
        model: nn.Module,
        decoders: Dict[str, nn.Module],
        discriminators: Dict[str, nn.Module],
        optimizer: torch.optim.Optimizer,
        optimizers_dsm: Dict[str, torch.optim.Optimizer],
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        task_criterion: nn.modules.loss._Loss,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        cfg: DictConfig,
        wandb_logging: bool,
        cuts: np.ndarray,
        aux_loss: torch.nn.modules.loss._Loss,
    ) -> None:
        super().__init__(
            model,
            discriminators,
            optimizer,
            optimizers_dsm,
            scheduler,
            task_criterion,
            dataloaders,
            cfg,
            wandb_logging,
            cuts,
            aux_loss,
        )
        self.decoders = decoders

    def shared_loop(self, split: str) -> Dict[str, float]:
        self.put_model_on_correct_mode(split)
        total_loss = 0.0
        embeddings = defaultdict(list)
        masks = defaultdict(list)
        total_sh_loss = 0.0
        total_u_loss = 0.0
        total_discriminators_loss = {k: 0.0 for k in self.optimizers_dsm.keys()}
        total_decoders_loss = {k: 0.0 for k in self.decoders.keys()}
        for batch in tqdm.tqdm(self.dataloaders[split]):
            data, time, _ = batch
            # check if data contains mask
            if isinstance(data, list):
                data, mask = data
                batch_embeddings_sh, batch_embeddings_u = self.model(
                    data, mask, return_embedding=True
                )
            else:
                batch_embeddings_sh, batch_embeddings_u = self.model(
                    data, return_embedding=True
                )

            sh_loss = self.aux_loss(
                {k: v.to(self.device) for k, v in batch_embeddings_sh.items()},
                {k: v.to(self.device) for k, v in mask.items()},
            )
            # put mask on cpu
            mask = {k: v.cpu() for k, v in mask.items()}
            x_r = {
                k: torch.cat(
                    [
                        batch_embeddings_sh[k][mask[k]]
                        .detach()
                        .to(batch_embeddings_u[k].device),
                        batch_embeddings_u[k][mask[k]],
                    ],
                    dim=1,
                )
                for k in batch_embeddings_u.keys()
            }
            x_r_prime = {
                k: torch.cat(
                    [
                        torch.roll(
                            batch_embeddings_sh[k][mask[k]]
                            .detach()
                            .to(batch_embeddings_u[k].device),
                            1,
                            0,
                        ),
                        batch_embeddings_u[k][mask[k]],
                    ],
                    dim=1,
                )
                for k in batch_embeddings_sh.keys()
            }
            u_loss = torch.tensor(0.0).to(self.device)
            decoder_loss_iteration = torch.tensor(0.0).to(self.device)
            for modality in x_r.keys():
                if sum(mask[modality]) <= 1:
                    continue
                fake_logits = self.discriminators[modality](x_r[modality])
                u_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                    input=fake_logits, target=torch.ones_like(fake_logits)
                ).to(self.device)
                decoder_outputs = self.decoders[modality](
                    batch_embeddings_u[modality][mask[modality]]
                )
                decoder_loss = torch.nn.functional.mse_loss(
                    decoder_outputs,
                    data[modality][mask[modality]].to(decoder_outputs.device),
                )
                decoder_loss_iteration += decoder_loss.to(self.device)
                total_decoders_loss[modality] += decoder_loss.item() * time.size(0)

            loss_encoders = (
                self.cfg.aux_loss.alpha * sh_loss
                + self.cfg.disentangled.gamma * u_loss
                + decoder_loss_iteration
            )
            if split == "train":
                self.optimizer.zero_grad()
                loss_encoders.backward()
                self.optimizer.step()
            else:
                for modality in batch_embeddings_sh.keys():
                    embeddings[modality].append(batch_embeddings_sh[modality].cpu())
                    masks[modality].append(mask[modality].cpu())

            losses_dsm = {}
            for modality in x_r.keys():
                if sum(mask[modality]) <= 1:
                    losses_dsm[modality] = 0.0
                    continue

                fake_logits = self.discriminators[modality](x_r[modality].detach())
                real_logits = self.discriminators[modality](
                    x_r_prime[modality].detach()
                )
                loss_dsm = self.discriminator_loss(
                    real_logits=real_logits, fake_logits=fake_logits
                )

                if split == "train":
                    self.optimizers_dsm[modality].zero_grad()
                    loss_dsm.backward()
                    self.optimizers_dsm[modality].step()

                total_discriminators_loss[modality] += loss_dsm.item()

            total_sh_loss += sh_loss.item() * time.size(0)
            total_u_loss += u_loss.item() * time.size(0)

        outputs = {
            "loss": total_loss / len(self.dataloaders[split].dataset),
            "sh_loss": total_sh_loss / len(self.dataloaders[split].dataset),
            "unique_loss": total_u_loss / len(self.dataloaders[split].dataset),
        }
        outputs.update(
            {
                f"discriminator_{k}": v / len(self.dataloaders[split].dataset)
                for k, v in total_discriminators_loss.items()
            }
        )
        outputs.update(
            {
                f"decoder_{k}": v / len(self.dataloaders[split].dataset)
                for k, v in total_decoders_loss.items()
            }
        )

        return outputs

    def finetune(self):
        self.shared_loop = self.shared_loop_finetune
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), **self.cfg.optimizer.params
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30, eta_min=5e-6
        )
        for name, param in self.model.named_parameters():
            if "_encoder" in name:
                param.requires_grad = False
        self.fit(epochs=self.cfg.general.epochs_finetune)

    def shared_loop_finetune(self, split: str) -> Dict[str, float]:
        self.put_model_on_correct_mode(split)
        total_loss = 0.0
        raw_predictions = []
        times = []
        events = []
        for batch in tqdm.tqdm(self.dataloaders[split]):
            data, time, event = batch
            # check if data contains mask
            if isinstance(data, list):
                data, mask = data
                outputs = self.model(data, mask, return_embedding=False)
            else:
                outputs = self.model(data, return_embedding=False)

            loss = self.task_criterion(
                outputs, time.to(outputs.device), event.to(outputs.device)
            )
            if split == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item() * time.size(0)
            raw_predictions.append(outputs)
            times.append(time)
            events.append(event)

        outputs = {"task_loss": total_loss / len(self.dataloaders[split].dataset)}
        if split != "train":
            task_metrics = self.compute_survival_metrics(
                raw_predictions, torch.cat(times, dim=0), torch.cat(events, dim=0)
            )
            outputs.update(task_metrics)

        return outputs

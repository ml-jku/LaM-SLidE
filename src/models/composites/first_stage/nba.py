from functools import partial
from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics import (
    AUROC,
    Accuracy,
    MeanMetric,
    MetricCollection,
    Precision,
    Recall,
)

from src.models.composites.lightning_base import BackboneBase, FirstStageLightningBase
from src.modules.losses import InterDistanceLoss, MaskedMSELoss, MaskedNormLoss


class Backbone(BackboneBase):

    def __init__(
        self,
        dim_input: int,
        dim_latent: int,
        encoder: nn.Module,
        decoder: nn.Module,
        embed_entity: nn.Module,
        embed_team: nn.Embedding,
        embed_group: nn.Embedding,
        act: nn.Module = partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__(
            dim_latent=dim_latent,
            encoder=encoder(entity_embedding=embed_entity),
            decoder=decoder(entity_embedding=embed_entity),
        )
        self.embed_entity: nn.Module = embed_entity
        self.embed_team: nn.Embedding = embed_team
        self.embed_group: nn.Embedding = embed_group

        dim_embed_team = embed_team.embedding_dim
        dim_embed_group = embed_group.embedding_dim

        self.net_merge = nn.Sequential(
            nn.Linear(dim_embed_team + dim_embed_group + 2, dim_input),
            act(),
            nn.Linear(dim_input, dim_input),
        )

    def prepare_inputs(self, batch: Dict[str, Tensor]) -> Tensor:
        team_embeddings = self.embed_team(batch["team"])
        group_embeddings = self.embed_group(batch["group"])
        x = torch.cat([batch["pos"], team_embeddings, group_embeddings], dim=-1)
        x = self.net_merge(x)
        return x


class Wrapper(FirstStageLightningBase):

    def __init__(
        self,
        backbone: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        scheduler: DictConfig = None,
        monitor: str = "val/loss",
        interval: str = "epoch",
        compile: bool = False,
        shift: float = 0.0,
        scale: float = 1.0,
        ema: DictConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.backbone: Backbone = instantiate(self.hparams.backbone)
        self.loss: Loss = instantiate(self.hparams.loss)

        if self.hparams.compile:
            self.backbone = torch.compile(self.backbone)

        if self.hparams.ema is not None:
            self.ema = instantiate(self.hparams.ema)(model=self)

        self._init_metrics()

    def _init_metrics(self) -> None:
        self.train_team_metrics = MetricCollection(
            {
                "auroc": AUROC(num_classes=3, task="multiclass"),
                "accuracy": Accuracy(num_classes=3, task="multiclass"),
                "precision": Precision(num_classes=3, task="multiclass"),
                "recall": Recall(num_classes=3, task="multiclass"),
            },
            prefix="train/team/",
        )
        self.val_team_metrics = self.train_team_metrics.clone(prefix="val/team/")

        self.train_group_metrics = MetricCollection(
            {
                "auroc": AUROC(num_classes=2, task="multiclass"),
                "accuracy": Accuracy(num_classes=2, task="multiclass"),
                "precision": Precision(num_classes=2, task="multiclass"),
                "recall": Recall(num_classes=2, task="multiclass"),
            },
            prefix="train/group/",
        )
        self.val_group_metrics = self.train_group_metrics.clone(prefix="val/group/")

        self.mean_metrics = nn.ModuleDict(
            {
                "loss": MeanMetric(),
                "pos_loss": MeanMetric(),
                "inter_distance_loss": MeanMetric(),
                "norm_loss": MeanMetric(),
                "team_loss": MeanMetric(),
                "group_loss": MeanMetric(),
                "dist": MeanMetric(),
            }
        )

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        loss, preds = self.model_step(batch)

        dataloader_name = self.trainer.datamodule.dataloader_names(dataloader_idx)
        self.log_dict(
            {f"train/{dataloader_name}/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["pos"].size(0),
            add_dataloader_idx=False,
        )

        team_preds_flat, group_preds_flat = map(
            lambda x: rearrange(x, "B M D -> (B M) D"),
            (preds["team"], preds["group"]),
        )
        team_targets_flat, group_targets_flat = map(
            lambda x: rearrange(x, "B M -> (B M)"),
            (batch["team"], batch["group"]),
        )
        self.train_team_metrics(team_preds_flat, team_targets_flat)
        self.train_group_metrics(group_preds_flat, group_targets_flat)
        self.log_dict(
            self.train_team_metrics, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            self.train_group_metrics, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        loss, preds = self.model_step(batch)

        dataloader_name = self.trainer.datamodule.dataloader_names(dataloader_idx)
        self.log_dict(
            {f"val/{dataloader_name}/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["pos"].size(0),
            add_dataloader_idx=False,
        )

        for k, v in loss.items():
            self.mean_metrics[k].update(v)

        team_preds_flat, group_preds_flat = map(
            lambda x: rearrange(x, "B M D -> (B M) D"),
            (preds["team"], preds["group"]),
        )
        team_targets_flat, group_targets_flat = map(
            lambda x: rearrange(x, "B M -> (B M)"),
            (batch["team"], batch["group"]),
        )
        self.val_team_metrics(team_preds_flat, team_targets_flat)
        self.val_group_metrics(group_preds_flat, group_targets_flat)
        self.log_dict(
            self.val_team_metrics, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            self.val_group_metrics, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True
        )
        return loss

    def on_validation_end(self):
        for k, v in self.mean_metrics.items():
            self.logger.experiment.log(
                {f"val/{k}": v.compute(), "step": self.global_step, "epoch": self.current_epoch}
            )
            v.reset()

    @torch.no_grad()
    def encode(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.backbone.encode(batch)

    def decode(self, latents: Tensor, entities: Tensor) -> Tensor:
        out = self.backbone.decode(
            z=latents,
            entities=entities,
        )
        return out

    @property
    def shift(self):
        return self.hparams.shift

    @property
    def scale(self):
        return self.hparams.scale


class Loss(nn.Module):

    def __init__(
        self,
        loss_pos_weight: float = 1.0,
        loss_inter_distance_weight: float = 1.0,
        loss_norm_weight: float = 0.0,
        loss_team_weight: float = 0.01,
        loss_group_weight: float = 0.01,
        loss_pos: nn.Module = MaskedMSELoss(),
        loss_inter_distance: nn.Module = InterDistanceLoss(),
        loss_norm: nn.Module = MaskedNormLoss(),
        loss_team: nn.Module = nn.CrossEntropyLoss(),
        loss_group: nn.Module = nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.loss_pos_weight = loss_pos_weight
        self.loss_inter_distance_weight = loss_inter_distance_weight
        self.loss_norm_weight = loss_norm_weight
        self.loss_team_weight = loss_team_weight
        self.loss_group_weight = loss_group_weight

        self.loss_pos = loss_pos
        self.loss_inter_distance = loss_inter_distance
        self.loss_norm = loss_norm
        self.loss_team = loss_team
        self.loss_group = loss_group

    def forward(self, model, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        mask = batch["attention_mask"]
        preds = model(batch)
        pos_preds = preds["pos"]
        pos_targets = batch["pos"]

        mask_flat = rearrange(mask, "B S -> (B S)")
        pos_preds_flat = rearrange(pos_preds, "B S D -> (B S) D")
        pos_targets_flat = rearrange(pos_targets, "B S D -> (B S) D")

        team_preds_flat, group_preds_flat = map(
            lambda x: rearrange(x, "B M D -> (B M) D"),
            (preds["team"], preds["group"]),
        )
        team_targets_flat, group_targets_flat = map(
            lambda x: rearrange(x, "B M -> (B M)"),
            (batch["team"], batch["group"]),
        )

        loss_team = self.loss_team(team_preds_flat, team_targets_flat)
        loss_group = self.loss_group(group_preds_flat, group_targets_flat)
        loss_pos = self.loss_pos(pos_preds_flat, pos_targets_flat, mask_flat)
        loss_inter_distance = self.loss_inter_distance(pos_preds, pos_targets, mask)
        loss_norm = self.loss_norm(pos_preds, pos_targets, mask)
        loss_team = self.loss_team(team_preds_flat, team_targets_flat)
        loss_group = self.loss_group(group_preds_flat, group_targets_flat)

        loss_total = (
            self.loss_pos_weight * loss_pos
            + self.loss_inter_distance_weight * loss_inter_distance
            + self.loss_norm_weight * loss_norm
            + self.loss_team_weight * loss_team
            + self.loss_group_weight * loss_group
        )

        return {
            "loss": loss_total,
            "pos_loss": loss_pos,
            "inter_distance_loss": loss_inter_distance,
            "norm_loss": loss_norm,
            "team_loss": loss_team,
            "group_loss": loss_group,
            "dist": loss_norm * model.scale,
        }, preds

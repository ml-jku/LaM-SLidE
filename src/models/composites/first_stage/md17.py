from functools import partial
from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics import MeanMetric

from src.models.composites.lightning_base import BackboneBase, FirstStageLightningBase
from src.modules.losses import (
    InterDistanceLoss,
    MaskedCrossEntropyLoss,
    MaskedMSELoss,
    MaskedNormLoss,
)


class Backbone(BackboneBase):

    def __init__(
        self,
        dim_input: int,
        dim_latent: int,
        encoder: nn.Module,
        decoder: nn.Module,
        embed_entity: nn.Module,
        embed_atom: nn.Module,
        embed_pos: nn.Module,
        act: nn.Module = partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__(
            dim_latent=dim_latent,
            encoder=encoder(entity_embedding=embed_entity),
            decoder=decoder(entity_embedding=embed_entity),
        )
        self.embed_entity: nn.Module = embed_entity
        self.embed_atom: nn.Module = embed_atom
        self.embed_pos: nn.Module = embed_pos

        dim_embed_atom = embed_atom.embedding_dim
        dim_embed_pos = embed_pos.embedding_dim

        self.net_merge = nn.Sequential(
            nn.Linear(dim_embed_atom + dim_embed_pos, dim_input),
            act(),
            nn.Linear(dim_input, dim_input),
        )

    def prepare_inputs(self, batch: Dict[str, Tensor]) -> Tensor:
        pos, atom = batch["pos"], batch["atom"]
        embed_atom = self.embed_atom(atom)
        embed_point = self.embed_pos(pos)
        x = torch.cat([embed_atom, embed_point], dim=-1)
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
        self.backbone: nn.Module = instantiate(self.hparams.backbone)
        self.loss: nn.Module = instantiate(self.hparams.loss)

        if self.hparams.compile:
            self.backbone = torch.compile(self.backbone)

        if self.hparams.ema is not None:
            self.ema = instantiate(self.hparams.ema)(model=self)

        self.mean_metrics = nn.ModuleDict(
            {
                "loss": MeanMetric(),
                "pos_loss": MeanMetric(),
                "atom_type_loss": MeanMetric(),
                "inter_distance_loss": MeanMetric(),
                "norm_loss": MeanMetric(),
                "dist": MeanMetric(),
            }
        )

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        loss, _ = self.model_step(batch)

        self.log_dict(
            {f"train/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["pos"].size(0),
        )
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        loss, _ = self.model_step(batch)

        dataloader_name = self.trainer.datamodule.dataloader_names(dataloader_idx)
        self.log_dict(
            {f"val/{dataloader_name}/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["pos"].size(0),
            add_dataloader_idx=False,
        )
        for name, value in loss.items():
            self.mean_metrics[name].update(value)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            {f"val/{k}": v.compute() for k, v in self.mean_metrics.items()},
            on_step=False,
            on_epoch=True,
        )
        for metric in self.mean_metrics.values():
            metric.reset()


class Loss(nn.Module):

    def __init__(
        self,
        loss_pos_weight: float = 1.0,
        loss_atom_type_weight: float = 0.0,
        loss_inter_distance_weight: float = 1.0,
        loss_norm_weight: float = 0.0,
        loss_pos: nn.Module = MaskedMSELoss(),
        loss_atom_type: nn.Module = MaskedCrossEntropyLoss(),
        loss_inter_distance: nn.Module = InterDistanceLoss(),
        loss_norm: nn.Module = MaskedNormLoss(),
    ):
        super().__init__()
        self.loss_pos_weight = loss_pos_weight
        self.loss_atom_type_weight = loss_atom_type_weight
        self.loss_inter_distance_weight = loss_inter_distance_weight
        self.loss_norm_weight = loss_norm_weight
        self.loss_pos = loss_pos
        self.loss_atom_type = loss_atom_type
        self.loss_inter_distance = loss_inter_distance
        self.loss_norm = loss_norm

    def forward(self, model, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        preds = model(batch)
        mask = batch["attention_mask"]
        pos_preds = preds["pos"]
        pos_targets = batch["pos"]
        atom_preds = preds["atom"]
        atom_targets = batch["atom"]

        mask_flat = rearrange(mask, "B S -> (B S)")
        pos_preds_flat = rearrange(pos_preds, "B S D -> (B S) D")
        pos_targets_flat = rearrange(pos_targets, "B S D -> (B S) D")
        atom_preds_flat = rearrange(atom_preds, "B S D -> (B S) D")
        atom_targets_flat = rearrange(atom_targets, "B S -> (B S)")

        loss_pos = self.loss_pos(pos_preds_flat, pos_targets_flat, mask_flat)
        loss_inter_distance = self.loss_inter_distance(pos_preds, pos_targets, mask)
        loss_norm = self.loss_norm(pos_preds, pos_targets, mask)

        loss_atom_type = self.loss_atom_type(
            logits=atom_preds_flat, target=atom_targets_flat, mask=mask_flat
        )

        loss_total = (
            self.loss_pos_weight * loss_pos
            + self.loss_inter_distance_weight * loss_inter_distance
            + self.loss_atom_type_weight * loss_atom_type
            + self.loss_norm_weight * loss_norm
        )

        return {
            "loss": loss_total,
            "pos_loss": loss_pos,
            "inter_distance_loss": loss_inter_distance,
            "atom_type_loss": loss_atom_type,
            "norm_loss": loss_norm,
            "dist": loss_norm * model.scale,
        }, preds

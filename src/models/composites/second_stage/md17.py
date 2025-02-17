from typing import Any, Dict, List

import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn
from torchmetrics import MeanMetric

from src.datasets.md17 import dataset_cond_indices
from src.models.composites.lightning_base import SecondStageCondLightningBase
from src.modules.losses import InterDistanceLoss, MaskedMSELoss, MaskedNormLoss
from src.modules.transport.transport import ModelType, Transport
from src.utils.utils import load_class


class Wrapper(SecondStageCondLightningBase):

    def __init__(
        self,
        n_atom_types: int,
        first_stage_model: DictConfig,
        backbone: DictConfig,
        transport: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        scheduler: DictConfig = None,
        monitor: str = "val/loss",
        interval: str = "epoch",
        compile: bool = False,
        cond_idx: List[int] = [0, 10],
        K: int = 5,
        sampling_method: str = "ODE",
        sampling_kwargs: Dict[str, Any] = {"sampling_method": "euler", "num_steps": 10},
        ema: DictConfig = None,
        mask_cond_mean: bool = False,
        num_timesteps: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.backbone: nn.Module = instantiate(self.hparams.backbone)
        self.si: Transport = instantiate(self.hparams.transport)()
        self.loss: Loss = instantiate(self.hparams.loss)

        self.first_stage_model = load_class(
            self.hparams.first_stage_model.class_name
        ).load_from_checkpoint(self.hparams.first_stage_model.path, map_location=self.device)
        self.first_stage_model.load_ema_weights()
        self.first_stage_model.eval()
        self.first_stage_model.freeze()

        if self.hparams.compile:
            self.first_stage_model = torch.compile(self.first_stage_model, fullgraph=True)
            self.backbone = torch.compile(self.backbone, fullgraph=True)

        self.ade_metric = nn.ModuleDict(
            {f"val/{dataset}/ade": MeanMetric() for dataset in dataset_cond_indices.keys()}
        )
        self.fde_metric = nn.ModuleDict(
            {f"val/{dataset}/fde": MeanMetric() for dataset in dataset_cond_indices.keys()}
        )

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss, _ = self.model_step(batch)

        self.log_dict(
            {f"train/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["x1"].size(0),
        )
        return loss

    @torch.no_grad()
    def validation_step(
        self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        loss, _ = self.model_step(batch)

        dataloader_name = self.trainer.datamodule.dataloader_names(dataloader_idx)
        preds_pos = self.sample(batch)["pos"]
        preds_pos = preds_pos[:, self.hparams.cond_idx[1] :]
        true_pos = batch["pos"][:, self.hparams.cond_idx[1] :].clone()
        ade = torch.norm(true_pos - preds_pos, dim=-1).mean(dim=(1, 2)) * self.scale
        fde = torch.norm(true_pos[:, -1] - preds_pos[:, -1], dim=-1).mean(dim=1) * self.scale

        self.ade_metric[f"val/{dataloader_name}/ade"](ade)
        self.fde_metric[f"val/{dataloader_name}/fde"](fde)

        self.log(
            f"val/{dataloader_name}/ade",
            self.ade_metric[f"val/{dataloader_name}/ade"],
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

        self.log(
            f"val/{dataloader_name}/fde",
            self.fde_metric[f"val/{dataloader_name}/fde"],
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

        self.log_dict(
            {f"val/{dataloader_name}/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["x1"].size(0),
            add_dataloader_idx=False,
        )

    @torch.no_grad()
    def encode(self, batch: Dict[str, Tensor]) -> Tensor:
        B = batch["entities"].shape[0]
        batch_enc = {
            k: rearrange(v, "B T ... -> (B T) ...")
            for k, v in batch.items()
            if k in ["pos", "atom", "attention_mask", "entities"]
        }
        latents = self.first_stage_model.encode(batch_enc)
        latents = rearrange(latents, "(B T) ... -> B T ...", B=B)
        return latents

    def decode(self, latents: Tensor, entities: Tensor) -> Tensor:
        preds = self.first_stage_model.decode(latents=latents, entities=entities)
        pos = rearrange(preds["pos"], "(B T) L D -> B T L D", T=self.hparams.num_timesteps)
        return {"pos": pos}

    @property
    def scale(self) -> float:
        return self.first_stage_model.hparams.scale

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = {}

    def test_step(
        self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """We run K times to get the mean ADE and FDE over multiple runs, to be consistent with
        GeoTDM paper, we set K=5."""
        dataloader_name = self.trainer.datamodule.dataloader_names(dataloader_idx)
        if dataloader_name not in self.test_step_outputs:
            self.test_step_outputs[dataloader_name] = {"ades": [], "fdes": []}

        true_pos = batch["pos"][:, self.hparams.cond_idx[1] :].clone()
        batch["pos"][:, self.hparams.cond_idx[1] :] = 0
        batch["vel"] = 0  # vel is not used anyway in our models
        batch["atom"][:, self.hparams.cond_idx[1] :] = 0

        # Sanity check to make sure target is not used in the conditioning.
        assert batch["pos"][:, self.hparams.cond_idx[1] :].sum() == 0
        assert batch["atom"][:, self.hparams.cond_idx[1] :].sum() == 0

        ades, fdes = [], []
        for _ in range(self.hparams.K):
            B = batch["pos"].shape[0]
            preds_pos = rearrange(self.sample(batch)["pos"], "(B T) L D -> B T L D", B=B)
            preds_pos = preds_pos[:, self.hparams.cond_idx[1] :]

            ade = torch.norm(true_pos - preds_pos, dim=-1).mean(dim=(1, 2))
            fde = torch.norm(true_pos[:, -1] - preds_pos[:, -1], dim=-1).mean(dim=1)
            ades.append(ade)
            fdes.append(fde)

        ades = torch.stack(ades).mean(dim=0)
        fdes = torch.stack(fdes).mean(dim=0)
        self.test_step_outputs[dataloader_name]["ades"].append(ades)
        self.test_step_outputs[dataloader_name]["fdes"].append(fdes)

    def on_test_epoch_end(self) -> Tensor:
        for dataloader_name, outputs in self.test_step_outputs.items():
            ades = torch.cat(outputs["ades"]) * self.scale
            fdes = torch.cat(outputs["fdes"]) * self.scale

            self.log(f"test/{dataloader_name}/ade", ades.mean())
            self.log(f"test/{dataloader_name}/fde", fdes.mean())


class CondWrapper(Wrapper):
    def __init__(self, n_classes: int = 1, vec_in_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.vec_in_embedding = nn.Embedding(n_classes, vec_in_dim)
        self.save_hyperparameters(logger=False)

    def prepare_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = super().prepare_batch(batch)
        batch["model_kwargs"]["y"] = self.vec_in_embedding(batch["cond_molecule"])
        return batch


class Loss(nn.Module):

    def __init__(
        self,
        weight_si_loss: float = 1.0,
        weight_pos_loss: float = 0.0,
        weight_inter_dist_loss: float = 0.0,
        weight_norm_loss: float = 0.0,
        loss_pos: nn.Module = MaskedMSELoss(),
        loss_inter_dist: nn.Module = InterDistanceLoss(),
        loss_norm: nn.Module = MaskedNormLoss(),
        calc_additional_losses: bool = False,
    ) -> None:
        super().__init__()
        self.weight_si_loss = weight_si_loss

        self.weight_pos_loss = weight_pos_loss
        self.weight_inter_dist_loss = weight_inter_dist_loss
        self.weight_norm_loss = weight_norm_loss

        self.loss_pos = loss_pos
        self.loss_inter_dist = loss_inter_dist
        self.loss_norm = loss_norm

        self.calc_additional_losses = calc_additional_losses

    def forward(self, model: nn.Module, batch: Dict[str, Tensor]) -> Tensor:
        out = model.si.training_losses(
            model=model, x1=batch["x1"], model_kwargs=batch["model_kwargs"]
        )
        pred_latent = out["pred"]
        si_loss = out["loss"].mean()

        losses = {}
        losses["si_loss"] = si_loss
        losses["loss"] = si_loss * self.weight_si_loss

        if self.calc_additional_losses:
            assert (
                model.si.model_type == ModelType.DATA
            ), "Additional losses are currently only supported for DATA model"
            pred_latent, entities = map(
                lambda x: rearrange(x, "B T ... -> (B T) ..."), (pred_latent, batch["entities"])
            )
            pred = model.decode(pred_latent, entities)
            mask_flat = rearrange(batch["attention_mask"], "B T L -> (B T L)")
            targets_pos_flat = rearrange(batch["pos"], "B T L D -> (B T L) D")
            preds_pos_flat = rearrange(pred["pos"], "B T L D -> (B T L) D")
            pos_loss = self.loss_pos(preds_pos_flat, targets_pos_flat, mask_flat)
            dist = self.loss_norm(preds_pos_flat, targets_pos_flat, mask_flat)

            inter_dist_loss = self.loss_inter_dist(
                rearrange(pred["pos"], "B T L D -> (B T) L D"),
                rearrange(batch["pos"], "B T L D -> (B T) L D"),
                rearrange(batch["attention_mask"], "B T L -> (B T) L"),
            )

            losses["pos_loss"] = pos_loss
            losses["inter_dist_loss"] = inter_dist_loss
            losses["dist"] = dist
            losses["loss"] += self.weight_pos_loss * pos_loss
            losses["loss"] += self.weight_inter_dist_loss * inter_dist_loss

        return losses, pred_latent

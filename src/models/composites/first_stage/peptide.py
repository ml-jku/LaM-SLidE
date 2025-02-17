from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall

from src.models.composites.lightning_base import BackboneBase, FirstStageLightningBase
from src.modules.embeddings import SinCosPositionalEmbedding1D
from src.modules.geometry import atom14_to_frames, get_chi_atom_indices
from src.modules.losses import MaskedCosineLoss, MaskedMSELoss, MaskedNormLoss
from src.utils import residue_constants as rc
from src.utils.rigid_utils import Rigid
from src.utils.tensor_utils import batched_gather


class Backbone(BackboneBase):

    def __init__(
        self,
        dim_input: int,
        dim_latent: int,
        encoder: nn.Module,
        decoder: nn.Module,
        embedding_entity: nn.Module,
        embedding_res: nn.Module,
        max_res: Optional[int] = None,
        act: nn.Module = partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__(
            dim_latent=dim_latent,
            encoder=encoder(entity_embedding=embedding_entity),
            decoder=decoder(entity_embedding=embedding_entity),
        )
        self.embedding_res: nn.Module = embedding_res

        dim_embed_res = embedding_res.embedding_dim

        self.embed_res_pos = (
            SinCosPositionalEmbedding1D(n_positions=max_res, embed_dim=dim_input)
            if max_res is not None
            else nn.Identity()
        )

        # 14 = Atom14 representation, 3 = x, y, z
        self.net_merge = nn.Sequential(
            nn.Linear(dim_embed_res + 14 * 3, dim_input),
            act(),
            nn.Linear(dim_input, dim_input),
        )

    # def forward(self, batch: Dict[str, Tensor]) -> Tensor:
    #     pos, res, entities = (
    #         batch["atom14_pos"],
    #         batch["aatype"],
    #         batch["entities"],
    #     )
    #     z = self.encode(
    #         pos=pos,
    #         res=res,
    #         entities=entities,
    #     )
    #     return self.decode(z=z, entities=entities)

    # def encode(self, pos: Tensor, res: Tensor, entities: Tensor, mask: Tensor = None) -> Tensor:
    #     x = self.prepare_inputs(pos=pos, res=res)
    #     latents = self.encoder(x, entities)
    #     z = self.quant(latents)
    #     return z

    def encode(self, batch: Dict[str, Tensor]) -> Tensor:
        x = self.prepare_inputs(batch)
        latents = self.encoder(x=x, entities=batch["entities"], mask=None)
        return self.quant(latents)

    # def decode(self, z: Tensor, entities: Tensor) -> Tensor:
    #     latents = self.post_quant(z)
    #     preds = self.decoder(latents, entities)
    #     preds["atom14_pos"] = rearrange(preds["atom14_pos"], "B R (A D) -> B R A D", A=14)
    #     return preds

    # def prepare_inputs(self, pos: Tensor, res: Tensor) -> Tensor:
    #     embed_res = self.embedding_res(res)
    #     pos = rearrange(pos, "B R A D -> B R (A D)")
    #     x = torch.cat([embed_res, pos], dim=-1)
    #     x = self.net_merge(x)
    #     x = self.embed_res_pos(x)
    #     return x

    def prepare_inputs(self, batch: Dict[str, Tensor]) -> Tensor:
        pos, res = batch["atom14_pos"], batch["aatype"]
        embed_res = self.embedding_res(res)
        pos = rearrange(pos, "B R A D -> B R (A D)")
        x = torch.cat([embed_res, pos], dim=-1)
        x = self.net_merge(x)
        x = self.embed_res_pos(x)
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
        self.loss: nn.Module = instantiate(self.hparams.loss, scale=self.scale)

        if self.hparams.compile:
            self.backbone = torch.compile(self.backbone)

        if self.hparams.ema is not None:
            self.ema = instantiate(self.hparams.ema)(model=self)

        self._init_metrics()

    def _init_metrics(self) -> None:
        self.train_metrics = MetricCollection(
            {
                "auroc": AUROC(num_classes=20, task="multiclass"),
                "accuracy": Accuracy(num_classes=20, task="multiclass"),
                "precision": Precision(num_classes=20, task="multiclass"),
                "recall": Recall(num_classes=20, task="multiclass"),
            },
            prefix="train/",
        )
        self.val_metrics = MetricCollection(
            {
                "auroc": AUROC(num_classes=20, task="multiclass"),
                "accuracy": Accuracy(num_classes=20, task="multiclass"),
                "precision": Precision(num_classes=20, task="multiclass"),
                "recall": Recall(num_classes=20, task="multiclass"),
            },
            prefix="val/",
        )

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        batch_size = batch["atom14_pos"].size(0)
        loss, outputs = self.model_step(batch)

        self.log_dict(
            {f"train/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        pred_aatype = rearrange(outputs["aatype"], "B R D -> (B R) D")
        target_aatype = rearrange(batch["aatype"], "B R -> (B R)")

        self.train_metrics(pred_aatype, target_aatype)
        self.log_dict(
            self.train_metrics, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        batch_size = batch["atom14_pos"].size(0)
        loss, outputs = self.model_step(batch)

        self.log_dict(
            {f"val/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        pred_aatype = rearrange(outputs["aatype"], "B R D -> (B R) D")
        target_aatype = rearrange(batch["aatype"], "B R -> (B R)")

        self.val_metrics(pred_aatype, target_aatype)
        self.log_dict(
            self.val_metrics, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True
        )

    @property
    def shift(self) -> float:
        return self.hparams.shift

    @property
    def scale(self) -> float:
        return self.hparams.scale

    # @torch.no_grad()
    # def encode(
    #     self,
    #     pos: Tensor,
    #     res: Tensor,
    #     entities: Tensor,
    #     mask: Tensor,
    # ) -> Tensor:
    #     return self.backbone.encode(
    #         pos=pos,
    #         res=res,
    #         entities=entities,
    #         mask=mask,
    #     )


class Loss(nn.Module):

    def __init__(
        self,
        loss_pos_weight: float = 1.0,
        loss_pos_frame_weight: float = 0.0,
        loss_res_type_weight: float = 0.0,
        loss_norm_weight: float = 0.0,
        loss_torsion_weight: float = 0.0,
        loss_inter_distance_weight: float = 0.0,
        loss_pos: nn.Module = MaskedMSELoss(),
        loss_pos_frame: nn.Module = MaskedMSELoss(),
        loss_res_type: nn.Module = CrossEntropyLoss(),
        loss_norm: nn.Module = MaskedNormLoss(),
        loss_torsion: nn.Module = MaskedCosineLoss(),
        loss_inter_distance: nn.Module = MaskedMSELoss(),
        scale: float = 1.0,
    ):
        super().__init__()
        self.loss_pos_weight = loss_pos_weight
        self.loss_pos_frame_weight = loss_pos_frame_weight
        self.loss_res_type_weight = loss_res_type_weight
        self.loss_norm_weight = loss_norm_weight
        self.loss_torsion_weight = loss_torsion_weight
        self.loss_inter_distance_weight = loss_inter_distance_weight
        self.loss_pos = loss_pos
        self.loss_pos_frame = loss_pos_frame
        self.loss_res_type = loss_res_type
        self.loss_norm = loss_norm
        self.loss_torsion = loss_torsion
        self.loss_inter_distance = loss_inter_distance
        self.scale = scale

        self.register_buffer(
            "restype_atom37_to_atom14",
            torch.tensor(rc.RESTYPE_ATOM37_TO_ATOM14),
            persistent=False,
        )
        self.register_buffer(
            "restype_atom14_mask",
            torch.tensor(rc.RESTYPE_ATOM37_MASK),
            persistent=False,
        )

    def atom14_to_atom37(
        self, atom14: Tensor, aatype: Tensor, atom14_mask: Tensor = None
    ) -> Tensor:
        """Same function as in geometry.py, but differentiable."""
        atom37 = batched_gather(
            atom14,
            self.restype_atom37_to_atom14[aatype],
            dim=-2,
            no_batch_dims=len(atom14.shape[:-2]),
        )
        atom37 *= self.restype_atom14_mask[aatype, :, None]
        if atom14_mask is not None:
            atom37_mask = batched_gather(
                atom14_mask,
                self.restype_atom37_to_atom14[aatype],
                dim=-1,
                no_batch_dims=len(atom14.shape[:-2]),
            )
            atom37_mask *= self.restype_atom14_mask[aatype]
            return atom37, atom37_mask
        else:
            return atom37

    def atom37_to_torsions(self, all_atom_positions, aatype, all_atom_mask=None):
        """Same function as in geometry.py, but differentiable."""
        if type(all_atom_positions) is np.ndarray:
            all_atom_positions = torch.from_numpy(all_atom_positions)
        if type(aatype) is np.ndarray:
            aatype = torch.from_numpy(aatype)
        if all_atom_mask is None:
            all_atom_mask = self.restype_atom14_mask[aatype]
        if type(all_atom_mask) is np.ndarray:
            all_atom_mask = torch.from_numpy(all_atom_mask)

        pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
        prev_all_atom_positions = torch.cat([pad, all_atom_positions[..., :-1, :, :]], dim=-3)

        pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
        prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

        pre_omega_atom_pos = torch.cat(
            [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
            dim=-2,
        )
        phi_atom_pos = torch.cat(
            [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
            dim=-2,
        )
        psi_atom_pos = torch.cat(
            [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
            dim=-2,
        )

        pre_omega_mask = torch.prod(prev_all_atom_mask[..., 1:3], dim=-1) * torch.prod(
            all_atom_mask[..., :2], dim=-1
        )
        phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
            all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
        )
        psi_mask = (
            torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
            * all_atom_mask[..., 4]
        )

        chi_atom_indices = torch.as_tensor(get_chi_atom_indices(), device=aatype.device)

        atom_indices = chi_atom_indices[..., aatype, :, :]
        chis_atom_pos = batched_gather(
            all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
        )

        chi_angles_mask = list(rc.chi_angles_mask)
        chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
        chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

        chis_mask = chi_angles_mask[aatype, :]

        chi_angle_atoms_mask = batched_gather(
            all_atom_mask,
            atom_indices,
            dim=-1,
            no_batch_dims=len(atom_indices.shape[:-2]),
        )
        chi_angle_atoms_mask = torch.prod(
            chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
        )
        chis_mask = chis_mask * chi_angle_atoms_mask

        torsions_atom_pos = torch.cat(
            [
                pre_omega_atom_pos[..., None, :, :],
                phi_atom_pos[..., None, :, :],
                psi_atom_pos[..., None, :, :],
                chis_atom_pos,
            ],
            dim=-3,
        )

        torsion_angles_mask = torch.cat(
            [
                pre_omega_mask[..., None],
                phi_mask[..., None],
                psi_mask[..., None],
                chis_mask,
            ],
            dim=-1,
        )

        torsion_frames = Rigid.from_3_points(
            torsions_atom_pos[..., 1, :],
            torsions_atom_pos[..., 2, :],
            torsions_atom_pos[..., 0, :],
            eps=1e-8,
        )

        fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

        torsion_angles_sin_cos = torch.stack(
            [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
        )

        denom = torch.sqrt(
            torch.sum(
                torch.square(torsion_angles_sin_cos),
                dim=-1,
                dtype=torsion_angles_sin_cos.dtype,
                keepdims=True,
            )
            + 1e-8
        )
        torsion_angles_sin_cos = torsion_angles_sin_cos / denom

        torsion_angles_sin_cos = (
            torsion_angles_sin_cos
            * all_atom_mask.new_tensor(
                [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
        )

        return torsion_angles_sin_cos, torsion_angles_mask

    def calc_torsions(self, atom14_pos: Tensor, aatype: Tensor) -> Tuple[Tensor, Tensor]:
        atom37_preds = self.atom14_to_atom37(atom14_pos, aatype)
        torsions_preds, _ = self.atom37_to_torsions(atom37_preds, aatype)
        return torsions_preds

    def forward(self, model, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        preds = model(batch)
        mask = batch["attention_mask"]  # TODO  Implement for variable length peptides.
        # We currently used only the tetrapeptide dataset for  so we dont need the mask currently because all peptides have the same length

        atom14_pos_targets = batch["atom14_pos"]
        atom14_pos_targets_flat = rearrange(atom14_pos_targets, "B R A D -> (B R A) D")
        atom14_pos_frame_targets = batch["atom14_pos_frame"]
        atom14_pos_frame_targets_flat = rearrange(atom14_pos_frame_targets, "B R A D -> (B R A) D")
        torsions_targets = batch["torsions"]

        torsions_mask = batch["torsions_mask"]
        aatype_targets = batch["aatype"]
        atom14_mask = batch["atom14_mask"]
        atom14_mask_flat = rearrange(atom14_mask, "... -> (...)")

        atom14_pos_preds = preds["atom14_pos"]
        atom14_frames_preds = atom14_to_frames(atom14_pos_preds).unsqueeze(-1)
        atom14_pos_frame_preds = atom14_frames_preds.invert_apply(atom14_pos_preds)
        atom14_pos_frame_preds_flat = rearrange(atom14_pos_frame_preds, "B R A D -> (B R A) D")
        aatype_preds = preds["aatype"]
        atom14_pos_preds_flat = rearrange(atom14_pos_preds, "B R A D -> (B R A) D")

        aatype_preds_flat = rearrange(aatype_preds, "B S D -> (B S) D")
        aatype_targets_flat = rearrange(aatype_targets, "B S -> (B S)")

        loss_pos = self.loss_pos(atom14_pos_preds_flat, atom14_pos_targets_flat, atom14_mask_flat)
        loss_pos_frame = self.loss_pos_frame(
            atom14_pos_frame_preds_flat, atom14_pos_frame_targets_flat, atom14_mask_flat
        )
        loss_inter_distance = self.loss_inter_distance(
            rearrange(atom14_pos_preds, "B R A D -> B (R A) D"),
            rearrange(atom14_pos_targets, "B R A D -> B (R A) D"),
            rearrange(atom14_mask, "B R A -> B (R A)"),
        )
        loss_norm = self.loss_norm(
            atom14_pos_preds_flat, atom14_pos_targets_flat, atom14_mask_flat
        )
        loss_res_type = self.loss_res_type(aatype_preds_flat, aatype_targets_flat)

        torsions_preds = self.calc_torsions(atom14_pos_preds, aatype_targets)
        torsions_preds_flat = rearrange(torsions_preds, "B R A D -> (B R A) D")
        torsions_targets_flat = rearrange(torsions_targets, "B R A D -> (B R A) D")
        torsions_mask_flat = rearrange(torsions_mask, "B R A -> (B R A)")

        loss_torsion = self.loss_torsion(
            preds=torsions_preds_flat,
            targets=torsions_targets_flat,
            mask=torsions_mask_flat,
        )

        loss_total = (
            self.loss_pos_weight * loss_pos
            + self.loss_pos_frame_weight * loss_pos_frame
            + self.loss_inter_distance_weight * loss_inter_distance
            + self.loss_res_type_weight * loss_res_type
            + self.loss_norm_weight * loss_norm
            + self.loss_torsion_weight * loss_torsion
        )

        return {
            "loss": loss_total,
            "pos_loss": loss_pos,
            "pos_frame_loss": loss_pos_frame,
            "inter_distance_loss": loss_inter_distance,
            "norm_loss": loss_norm,
            "res_type_loss": loss_res_type,
            "torsion_loss": loss_torsion,
            "dist": loss_norm * model.scale,
        }, preds

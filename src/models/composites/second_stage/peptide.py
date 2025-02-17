from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor

from src.models.composites.lightning_base import SecondStageCondLightningBase
from src.modules.geometry import atom14_to_frames, get_chi_atom_indices
from src.modules.losses import MaskedCosineLoss, MaskedMSELoss, MaskedNormLoss
from src.modules.transport.transport import ModelType, Transport
from src.utils import residue_constants as rc
from src.utils.rigid_utils import Rigid
from src.utils.tensor_utils import batched_gather
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
        n_timesteps: int = 100,
        sampling_method: str = "ODE",
        sampling_kwargs: Dict[str, Any] = {"sampling_method": "euler", "num_steps": 10},
        ema: DictConfig = None,
        mask_cond_mean: bool = False,
        self_optimization_prob: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.backbone: nn.Module = instantiate(self.hparams.backbone)
        self.si: Transport = instantiate(self.hparams.transport)()
        self.loss: Loss = instantiate(self.hparams.loss)

        self.first_stage_model = load_class(
            self.hparams.first_stage_model.class_name
        ).load_from_checkpoint(self.hparams.first_stage_model.path, map_location=self.device)
        if self.first_stage_model.ema is not None:
            self.first_stage_model.load_ema_weights()
        self.first_stage_model.eval()
        self.first_stage_model.freeze()

        if self.hparams.compile:
            self.first_stage_model = torch.compile(self.first_stage_model, fullgraph=True)
            self.backbone = torch.compile(self.backbone, fullgraph=True)

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
    def validation_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss, _ = self.model_step(batch)

        self.log_dict(
            {f"val/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["x1"].size(0),
        )
        return loss

    @torch.no_grad()
    def encode(self, batch: Dict[str, Tensor]) -> Tensor:
        B = batch["entities"].shape[0]
        batch_enc = {
            k: rearrange(v, "B T ... -> (B T) ...")
            for k, v in batch.items()
            if k in ["atom14_pos", "aatype", "attention_mask", "entities"]
        }
        latents = self.first_stage_model.encode(batch_enc)
        latents = rearrange(latents, "(B T) ... -> B T ...", B=B)
        return latents

    def decode(self, latents: Tensor, entities: Tensor) -> Tensor:
        preds = self.first_stage_model.decode(latents=latents, entities=entities)
        pos = rearrange(
            preds["atom14_pos"], "(B T) L (A D) -> B T L A D", T=self.hparams.n_timesteps, A=14
        )
        return {"atom14_pos": pos}


class Loss(nn.Module):

    def __init__(
        self,
        loss_si_weight: float = 1.0,
        loss_pos_weight: float = 1.0,
        loss_pos_frame_weight: float = 0.0,
        loss_norm_weight: float = 0.0,
        loss_torsion_weight: float = 0.0,
        loss_inter_distance_weight: float = 0.0,
        loss_pos: nn.Module = MaskedMSELoss(),
        loss_pos_frame: nn.Module = MaskedMSELoss(),
        loss_norm: nn.Module = MaskedNormLoss(),
        loss_torsion: nn.Module = MaskedCosineLoss(),
        loss_inter_distance: nn.Module = MaskedMSELoss(),
        calc_additional_losses: bool = False,
    ):
        super().__init__()
        self.loss_si_weight = loss_si_weight
        self.loss_pos_weight = loss_pos_weight
        self.loss_pos_frame_weight = loss_pos_frame_weight
        self.loss_norm_weight = loss_norm_weight
        self.loss_torsion_weight = loss_torsion_weight
        self.loss_inter_distance_weight = loss_inter_distance_weight
        self.loss_pos = loss_pos
        self.loss_pos_frame = loss_pos_frame
        self.loss_norm = loss_norm
        self.loss_torsion = loss_torsion
        self.loss_inter_distance = loss_inter_distance
        self.calc_additional_losses = calc_additional_losses

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
        out = model.si.training_losses(
            model=model, x1=batch["x1"], model_kwargs=batch["model_kwargs"]
        )
        pred_latent = out["pred"]
        si_loss = out["loss"].mean()

        losses = {}
        losses["si_loss"] = si_loss
        losses["loss"] = si_loss * self.loss_si_weight

        if self.calc_additional_losses:
            assert (
                model.si.model_type == ModelType.DATA
            ), "Additional losses are currently only supported for DATA model"
            pred_latent, entities = map(
                lambda x: rearrange(x, "B T ... -> (B T) ..."), (pred_latent, batch["entities"])
            )
            pred = model.decode(pred_latent, entities)

            _ = batch["attention_mask"]  # TODO  Implement for variable length peptides.
            # We currently used only the tetrapeptide dataset for  so we dont need the mask currently because all peptides have the same length

            atom14_pos_targets = batch["atom14_pos"]
            _, T, _, A, _ = atom14_pos_targets.size()

            atom14_pos_targets_flat = rearrange(atom14_pos_targets, "B T R A D -> (B T R A) D")
            atom14_pos_frame_targets_flat = rearrange(
                batch["atom14_pos_frame"], "B T R A D -> (B T R A) D"
            )
            torsions_targets = batch["torsions"]

            torsions_mask = batch["torsions_mask"]
            aatype_targets = batch["aatype"]
            atom14_mask = batch["atom14_mask"]
            atom14_mask_flat = rearrange(atom14_mask, "... -> (...)")

            atom14_pos_preds = rearrange(pred["atom14_pos"], "B T R A D -> (B T) R A D")
            atom14_frames_preds = atom14_to_frames(atom14_pos_preds).unsqueeze(-1)
            atom14_pos_preds_flat = rearrange(atom14_pos_preds, "(B T) R A D -> (B T R A) D", T=T)
            atom14_pos_frame_preds = atom14_frames_preds.invert_apply(atom14_pos_preds)
            atom14_pos_frame_preds_flat = rearrange(
                atom14_pos_frame_preds, "(B T) R A D -> (B T R A) D", T=T
            )

            loss_pos = self.loss_pos(
                atom14_pos_preds_flat, atom14_pos_targets_flat, atom14_mask_flat
            )
            loss_pos_frame = self.loss_pos_frame(
                atom14_pos_frame_preds_flat, atom14_pos_frame_targets_flat, atom14_mask_flat
            )
            loss_inter_distance = self.loss_inter_distance(
                rearrange(atom14_pos_preds, "(B T) R A D -> (B T) (R A) D", T=T),
                rearrange(atom14_pos_targets, "B T R A D -> (B T) (R A) D"),
                rearrange(atom14_mask, "B T R A -> (B T) (R A)"),
            )
            loss_norm = self.loss_norm(
                atom14_pos_preds_flat, atom14_pos_targets_flat, atom14_mask_flat
            )

            torsions_preds = self.calc_torsions(
                atom14_pos_preds, rearrange(aatype_targets, "B T R -> (B T) R")
            )
            torsions_preds_flat = rearrange(torsions_preds, "(B T) R A D -> (B T R A) D", T=T)
            torsions_targets_flat = rearrange(torsions_targets, "B T R A D -> (B T R A) D")
            torsions_mask_flat = rearrange(torsions_mask, "B T R A -> (B T R A)")

            loss_torsion = self.loss_torsion(
                preds=torsions_preds_flat,
                targets=torsions_targets_flat,
                mask=torsions_mask_flat,
            )

            losses["pos_loss"] = loss_pos
            losses["pos_frame_loss"] = loss_pos_frame
            losses["inter_distance_loss"] = loss_inter_distance
            losses["norm_loss"] = loss_norm
            losses["torsion_loss"] = loss_torsion

            losses["loss"] += self.loss_pos_weight * loss_pos
            losses["loss"] += self.loss_pos_frame_weight * loss_pos_frame
            losses["loss"] += self.loss_inter_distance_weight * loss_inter_distance
            losses["loss"] += self.loss_norm_weight * loss_norm
            losses["loss"] += self.loss_torsion_weight * loss_torsion

        return losses, pred_latent

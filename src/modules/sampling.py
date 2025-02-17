from typing import Dict

import mdtraj as md
import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm

import src.utils.residue_constants as rc
from src.modules.geometry import atom14_to_atom37
from src.utils.residue_constants import restype_atom14_mask, restype_order
from src.utils.traj_utils import traj_to_atom14


class SIAtom14SamplingWrapper:

    def __init__(
        self,
        model,
    ):
        self.model = model

    def create_batch(
        self,
        pos: Tensor,  # [R, 3]
        res: Tensor,  # [R]
        res_mask: Tensor,  # [R]
    ) -> Dict[str, Tensor]:
        T = self.model.hparams.n_timesteps
        pos = pos * res_mask[..., None].to(pos.device)
        batch = {
            "atom14_pos": repeat(pos, "R A D -> 1 T R A D", T=T),
            "aatype": repeat(res, "R -> 1 T R", T=T),
            "attention_mask": repeat(
                torch.ones_like(res, dtype=torch.bool),
                "R -> 1 T R",
                T=T,
            ),
            "entities": torch.arange(res.shape[0]).expand(T, -1).unsqueeze(0),
        }
        return batch

    def sample_rollout(
        self,
        cond_pos: Tensor,
        res: Tensor,
        res_mask: Tensor,
        num_rollouts: int = 1,
    ) -> Tensor:
        cond_pos = (cond_pos - self.model.shift) / self.model.scale
        pos = cond_pos.clone()

        rollouts = []
        for _ in tqdm(range(num_rollouts)):
            batch = self.create_batch(pos=pos, res=res, res_mask=res_mask)
            pred_pos = self.model.sample(batch)["atom14_pos"].squeeze(0)
            rollouts.append(pred_pos)
            pos = pred_pos[-1].clone()

        positions = torch.cat(rollouts)
        positions[0] = cond_pos
        return positions * self.model.scale + self.model.shift

    def sample_traj(
        self,
        traj: md.Trajectory,
        cond_frame_idx: int = 0,
        num_rollouts: int = 1,
    ) -> md.Trajectory:
        """Sample a trajectory from the model. To sample we need a single frame from a trajectory.

        Args:
            traj: Trajectory object.
            cond_frame_idx: Index of the frame to condition on.
            num_rollouts: Number of rollouts to sample.

        Returns:
            A trajectory object containing the sampled positions.
        """
        traj = traj.center_coordinates()
        traj = traj.atom_slice(traj.topology.select("not element H"))
        cond_pos = traj_to_atom14(traj[cond_frame_idx])[0]

        res = np.array([restype_order[resi.code] for resi in traj.top.residues])
        res_mask = torch.from_numpy(restype_atom14_mask[res]).to(torch.bool)

        cond_pos = torch.from_numpy(cond_pos).to(torch.float32)
        res = torch.from_numpy(res).to(torch.long)

        preds_pos = self.sample_rollout(cond_pos, res, res_mask, num_rollouts)
        preds_pos = preds_pos.detach().cpu().numpy()
        res = res.detach().cpu().numpy()
        res_mask = res_mask.detach().cpu().numpy()

        preds_pos = preds_pos * res_mask[None, ..., None]
        preds_pos = rearrange(preds_pos, "T R A D -> T R A D", A=14)

        traj_pred = atom14_to_mdtraj(preds_pos, res)
        return traj_pred


def atom14_to_mdtraj(atom14, aatype):
    """Convert atom14 positions to MDTraj trajectory format.

    Args:
        atom14: [N_frames, N_res, 14, 3] atom positions
        aatype: [N_res] residue types
    Returns:
        mdtraj.Trajectory: Trajectory object containing the frames
    """

    frames = []
    for pos in atom14:
        pos37 = atom14_to_atom37(pos, aatype)
        frames.append(pos37)

    xyz = np.stack(frames, axis=0)  # [n_frames, n_res, 37, 3]

    top = md.Topology()
    chain = top.add_chain()
    for _, aa in enumerate(aatype):
        resname = rc.restype_1to3[rc.restypes[aa]]
        residue = top.add_residue(resname, chain)

        atom_mask = rc.RESTYPE_ATOM37_MASK[aa]
        for atom_idx, present in enumerate(atom_mask):
            if present:
                atom_name = rc.atom_types[atom_idx]
                element = md.element.get_by_symbol(atom_name[0])
                top.add_atom(atom_name, element, residue)

    xyz_masked = []
    atom_idx = 0
    for aa in aatype:
        mask = rc.RESTYPE_ATOM37_MASK[aa]
        xyz_masked.append(xyz[:, atom_idx, mask.astype(bool), :])
        atom_idx += 1

    xyz_masked = np.concatenate(xyz_masked, axis=1)
    traj = md.Trajectory(xyz_masked, top)
    return traj

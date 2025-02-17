import os

import lightning as L
import mdtraj as md
import numpy as np
import torch
from einops import rearrange
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.datasets.collate_functions import CollatePadBatch, CollatePadBatchTempV2
from src.modules.geometry import atom14_to_atom37, atom14_to_frames, atom37_to_torsions
from src.utils import RankedLogger
from src.utils.data_utils import centre_random_augmentation, random_rotation_matrix
from src.utils.residue_constants import restype_atom14_mask, restype_order
from src.utils.traj_utils import load_traj, traj_to_atom14

log = RankedLogger(__name__, rank_zero_only=True)


class PeptideDataset(Dataset):
    def __init__(
        self,
        first_stage: bool,
        data_dir: str,
        rand_rotation: bool = False,
        rand_translation: float = 0,
        num_entities: int = 100,
        n_timesteps: int = 100,
        scale: float = 2.2,
        shift: float = 0,
        n_jobs: int = 1,
    ):
        super().__init__()
        self.first_stage = first_stage
        self.data_dir = data_dir
        self.rand_rotation = rand_rotation
        self.rand_translation = rand_translation
        self.num_entities = num_entities
        self.n_timesteps = n_timesteps
        self.scale = scale
        self.shift = shift
        self.trajectories = []
        self.valid_lengths = []

        self.samples = []
        unique_peptides = [
            f.split("-")[0] for f in os.listdir(self.data_dir) if f.endswith(".npz")
        ]
        unique_peptides.sort()

        if os.getenv("DEBUG"):
            unique_peptides = unique_peptides[:10]

        def process_trajectory(aa):
            traj_file = os.path.join(self.data_dir, f"{aa}-traj-arrays.npz")
            top_file = os.path.join(self.data_dir, f"{aa}-traj-state0.pdb")
            traj: md.Trajectory = load_traj(traj_file, top_file)
            traj.superpose(traj)
            traj.center_coordinates()

            atom14_pos = (traj_to_atom14(traj) - self.shift) / self.scale
            seqres = torch.tensor([restype_order[resi.code] for resi in traj.top.residues])
            aatype = seqres[None].expand(atom14_pos.shape[0], -1)
            frames = atom14_to_frames(torch.from_numpy(atom14_pos)).unsqueeze(-1)
            atom37 = torch.from_numpy(atom14_to_atom37(atom14_pos, aatype))

            torsions, torsions_mask = atom37_to_torsions(atom37, aatype)
            torsions = torch.nan_to_num(torsions)
            torsions = torsions * torsions_mask[..., None]
            atom14_pos_frame = frames.invert_apply(torch.from_numpy(atom14_pos))

            atom14_pos = torch.from_numpy(atom14_pos).to(torch.float32)
            atom14_pos = rearrange(atom14_pos, "T R A D -> T R A D")

            atom14_mask = restype_atom14_mask[aatype]

            n_frames = atom14_pos.shape[0]
            valid_length = n_frames - self.n_timesteps - 1
            assert valid_length > 0
            return {
                "atom14_pos": atom14_pos,
                "atom14_mask": torch.from_numpy(atom14_mask).to(torch.bool),
                "atom14_pos_frame": atom14_pos_frame,
                "torsions": torsions,
                "torsions_mask": torsions_mask,
                "aatype": aatype.to(torch.long),
                "trajectory_name": aa,
                "n_frames": n_frames,
                "top_file_path": top_file,
            }

        self.trajectories = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_trajectory)(p)
            for p in tqdm(
                unique_peptides,
                desc="Loading trajectories",
                dynamic_ncols=True,
            )
        )

        if not self.trajectories:
            raise ValueError("No valid trajectories found.")

    def augment_data(self, pos):
        rotation_matrix = random_rotation_matrix() if self.rand_rotation else torch.eye(3)
        translation_vector = torch.randn(3) * self.rand_translation
        return centre_random_augmentation(pos, R=rotation_matrix, translation=translation_vector)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index out of bounds.")
        traj = self.trajectories[idx]
        entities = torch.randperm(self.num_entities)[: traj["aatype"].shape[1]]

        if self.first_stage:
            rand_idx = np.random.randint(traj["n_frames"])
            atom14_pos = traj["atom14_pos"][rand_idx]
            atom14_mask = traj["atom14_mask"][rand_idx]
            atom14_pos_frame = traj["atom14_pos_frame"][rand_idx]
            aatype = traj["aatype"][rand_idx]
            torsions = traj["torsions"][rand_idx]
            torsions_mask = traj["torsions_mask"][rand_idx]

            R = atom14_pos.shape[0]
            atom14_pos = rearrange(atom14_pos, "R A D -> (R A) D")
            atom14_pos = self.augment_data(atom14_pos)
            atom14_pos = rearrange(atom14_pos, "(R A) D -> R A D", R=R)

        else:
            valid_frames = traj["n_frames"] - self.n_timesteps
            assert (
                valid_frames > 0
            ), f"Trajectory {traj['trajectory_name']} has insufficient frames"
            frame_start = np.random.randint(valid_frames)
            frame_end = frame_start + self.n_timesteps

            atom14_pos = traj["atom14_pos"][frame_start:frame_end]
            atom14_mask = traj["atom14_mask"][frame_start:frame_end]
            atom14_pos_frame = traj["atom14_pos_frame"][frame_start:frame_end]
            aatype = traj["aatype"][frame_start:frame_end]
            torsions = traj["torsions"][frame_start:frame_end]
            torsions_mask = traj["torsions_mask"][frame_start:frame_end]

            _, R, _, _ = atom14_pos.size()
            atom14_pos = rearrange(atom14_pos, "T R A D -> T (R A) D")
            atom14_pos = self.augment_data(atom14_pos)
            atom14_pos = rearrange(atom14_pos, "T (R A) D -> T R A D", R=R)
            entities = entities.expand(self.n_timesteps, -1)

        atom14_pos = atom14_pos * atom14_mask[..., None]
        return {
            "atom14_pos": atom14_pos,
            "atom14_mask": atom14_mask,
            "atom14_pos_frame": atom14_pos_frame,
            "aatype": aatype,
            "torsions": torsions,
            "torsions_mask": torsions_mask,
            "trajectory_name": traj["trajectory_name"],
            "entities": entities,
        }


class PeptideDataModule(L.LightningDataModule):

    def __init__(
        self,
        first_stage: bool,
        data_dir: str,
        rand_rotation: bool,
        rand_translation: float,
        num_entities: int,
        n_timesteps: int,
        shift: float,
        scale: float,
        batch_size: int,
        drop_last: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int,
        n_jobs: int = 1,
    ):
        super().__init__()
        self.first_stage = first_stage
        self.data_dir = data_dir
        self.rand_rotation = rand_rotation
        self.rand_translation = rand_translation
        self.num_entities = num_entities
        self.shift = shift
        self.scale = scale
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.n_jobs = n_jobs
        self.collate_fn = CollatePadBatch() if self.first_stage else CollatePadBatchTempV2()

    def _create_dataloader(self, mode: str, data_dir: str):
        return DataLoader(
            PeptideDataset(
                first_stage=self.first_stage,
                data_dir=data_dir,
                rand_rotation=self.rand_rotation if mode == "train" else False,
                rand_translation=self.rand_translation if mode == "train" else 0,
                num_entities=self.num_entities,
                n_timesteps=self.n_timesteps,
                shift=self.shift,
                scale=self.scale,
                n_jobs=self.n_jobs,
            ),
            batch_size=self.batch_size,
            shuffle=mode == "train",
            drop_last=mode == "train" and self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        data_dir = f"{self.data_dir}/train"
        return self._create_dataloader(mode="train", data_dir=data_dir)

    def val_dataloader(self):
        data_dir = f"{self.data_dir}/val"
        return self._create_dataloader(mode="val", data_dir=data_dir)

    def test_dataloader(self):
        data_dir = f"{self.data_dir}/test"
        return self._create_dataloader(mode="test", data_dir=data_dir)

# We use the split from https://github.com/xupei0610/SocialVAE/blob/main/data.py
# but process the data in a different way.
# We dont view each player as a separate trajectory, but one frame with all players
# as a single sample. SocialVAE treats each player as a separate trajectory, but the
# other players as neighbors.

import os
from bisect import bisect_right
from itertools import accumulate

import lightning as L
import numpy as np
import torch
from easydict import EasyDict as edict
from einops import rearrange
from joblib import Parallel, delayed
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from src.datasets.collate_functions import CollatePadBatch, CollatePadBatchTemp
from src.utils import RankedLogger
from src.utils.data_utils import random_rotation_matrix_2D, rotate_point_cloud

log = RankedLogger(__name__, rank_zero_only=True)

dataset_cond_indices: dict[str, int] = {
    "score": 0,
    "rebound": 1,
}


class NBADataset(Dataset):

    def __init__(
        self,
        first_stage: bool,
        data_dir: str,
        scene: str,
        num_frames: int = 20,
        flip: bool = False,
        rand_rotation: bool = False,
        rand_translation: float = 0,
        shift: float = 0,
        scale: float = 1,
        num_entities: int = 11,
        n_jobs: int = 60,
    ):
        super().__init__()
        self.first_stage = first_stage
        self.data_dir = data_dir
        self.scene = scene
        self.num_frames = num_frames
        self.flip = flip
        self.rand_rotation = rand_rotation
        self.rand_translation = rand_translation
        self.shift = shift
        self.scale = scale
        self.num_entities = num_entities
        self.n_jobs = n_jobs

        self.data = []
        files = sorted(os.listdir(self.data_dir))

        if os.getenv("DEBUG"):
            files = files[:10]

        self.data = Parallel(n_jobs=n_jobs)(
            delayed(self.process_file)(file) for file in tqdm(files, desc="Loading data")
        )

        self.valid_lengths = [0]
        for data in self.data:
            valid_length = data.pos.shape[0] - self.num_frames + 1
            self.valid_lengths.append(valid_length)
        self.cumulative_sizes = list(accumulate(self.valid_lengths))
        self.cond_index = torch.tensor([dataset_cond_indices[self.scene]], dtype=torch.long)

    def process_file(self, name):
        data = edict(np.load(f"{self.data_dir}/{name}"))
        data.name = name
        data.pos = (data.pos - self.shift) / self.scale
        data.pos = torch.from_numpy(data.pos).float()
        data.team = torch.from_numpy(data.team).long()
        data.group = torch.from_numpy(data.group).long()
        data.agent_id = torch.from_numpy(data.agent_id).long()
        if data.pos.shape[0] < self.num_frames:
            log.warning(f"Skipping {name} because it has less than {self.num_frames} frames")
            return None
        return data

    def __len__(self):
        if self.first_stage:
            return len(self.data)
        else:
            return self.cumulative_sizes[-1]

    def augment_data(self, pos, team):
        # Flip team, so that team embedding is not biased on the order.
        if self.flip:
            if np.random.rand() < 0.5:
                team[..., 1:5] = 2
                team[..., 6:] = 1

        rotation_matrix = random_rotation_matrix_2D() if self.rand_rotation else torch.eye(2)
        translation_vector = torch.randn(2) * self.rand_translation
        pos = rotate_point_cloud(pos, rotation_matrix) + translation_vector
        return pos, team

    def __getitem__(self, idx):

        if self.first_stage:
            traj_idx = np.random.randint(len(self.data))
            traj = self.data[traj_idx]
            rand_idx = np.random.randint(traj.pos.shape[0])
            pos = traj.pos[rand_idx]
            team = traj.team[rand_idx]
            group = traj.group[rand_idx]
            agent_id = traj.agent_id[rand_idx]
            pos, team = self.augment_data(pos, team)

            entities = torch.randperm(self.num_entities)[: pos.shape[0]].long()
            return {
                "pos": pos,
                "team": team,
                "group": group,
                "agent_id": agent_id,
                "entities": entities,
            }
        else:
            traj_idx = bisect_right(self.cumulative_sizes, idx)
            traj = self.data[traj_idx - 1]
            sample_idx = idx - self.cumulative_sizes[traj_idx - 1]
            pos = traj.pos[sample_idx : sample_idx + self.num_frames]
            team = traj.team[sample_idx : sample_idx + self.num_frames]
            group = traj.group[sample_idx : sample_idx + self.num_frames]
            agent_id = traj.agent_id[sample_idx : sample_idx + self.num_frames]

            T = pos.size(0)
            pos = rearrange(pos, "T A D -> (T A) D")
            pos, team = self.augment_data(pos, team)
            pos = rearrange(pos, "(T A) D -> T A D", T=T)
            entities = torch.randperm(self.num_entities)[: pos.shape[1]].long()
            entities = entities.expand(T, -1)
            return {
                "pos": pos.unsqueeze(0),
                "team": team.unsqueeze(0),
                "group": group.unsqueeze(0),
                "agent_id": agent_id.unsqueeze(0),
                "entities": entities.unsqueeze(0),
                "cond_scene": self.cond_index,
            }


class NBADatamodule(L.LightningDataModule):

    def __init__(
        self,
        first_stage: str,
        scenes: list[str],
        data_dir: str,
        num_frames: int = 20,
        flip: bool = False,
        rand_rotation: bool = False,
        rand_translation: float = 0,
        shift: float = 0,
        scale: float = 1,
        num_entities: int = 11,
        batch_size: int = 1,
        drop_last: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = None,
        n_jobs: int = 1,
    ):
        super().__init__()
        self.first_stage = first_stage
        self.scenes = scenes
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.flip = flip
        self.rand_rotation = rand_rotation
        self.rand_translation = rand_translation
        self.shift = shift
        self.scale = scale
        self.num_entities = num_entities
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_jobs = n_jobs
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.collate_fn = CollatePadBatch() if self.first_stage else CollatePadBatchTemp()

    def dataloader_names(self, idx: int):
        return self.scenes[idx]

    def _create_dataloader(self, mode, scenes: list[str]):
        return DataLoader(
            ConcatDataset(
                [
                    NBADataset(
                        first_stage=self.first_stage,
                        data_dir=f"{self.data_dir}/{scene}/{mode}",
                        scene=scene,
                        num_frames=self.num_frames,
                        flip=self.flip if mode == "train" else False,
                        rand_rotation=self.rand_rotation if mode == "train" else False,
                        rand_translation=self.rand_translation if mode == "train" else 0,
                        shift=self.shift,
                        scale=self.scale,
                        num_entities=self.num_entities,
                        n_jobs=self.n_jobs,
                    )
                    for scene in scenes
                ]
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=mode == "train" and self.drop_last,
            shuffle=mode == "train",
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        """For training, we use all scenes."""
        return self._create_dataloader("train", self.scenes)

    def val_dataloader(self):
        """For validation, we use each scene separately.

        Unfortunately this setup does not provide a validation set. Compared methods take the best
        model on the test set, which is in general not recommended, but to stay comparable with
        other methods we use the same setup.
        """
        return [self._create_dataloader(mode="test", scenes=[scene]) for scene in self.scenes]

    def test_dataloader(self):
        """For testing, we use each scene separately."""
        return [self._create_dataloader(mode="test", scenes=[scene]) for scene in self.scenes]

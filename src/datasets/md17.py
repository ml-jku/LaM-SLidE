from typing import Optional

import lightning as L
import torch
from einops import rearrange
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.datasets.collate_functions import CollatePadBatch, CollatePadBatchTemp
from src.datasets.geo_tdm.md17 import MD17Traj
from src.utils.data_utils import random_rotation_matrix, rotate_point_cloud

# This have to stay ordered, that conditioning index stays consistent,
# although if we do not train on all molecules, we can just ignore the ones
# that are not used.
dataset_cond_indices: dict[str, int] = {
    "aspirin": 0,
    "benzene": 1,
    "ethanol": 2,
    "malonaldehyde": 3,
    "naphthalene": 4,
    "salicylic": 5,
    "toluene": 6,
    "uracil": 7,
}


class MD17Dataset(Dataset):
    """Dataset for MD17.

    This is a wrapper around the implementation from
    https://github.com/hanjq17/GeoTDM/tree/main.
    """

    def __init__(
        self,
        first_stage: bool,
        root: str,
        molecule_name: str,
        with_h: bool,
        down_sample_every: int,
        span: int,
        charge_power: int = 1,
        force_reprocess: bool = False,
        force_length=None,
        mode=None,
        return_index=False,
        project=False,
        rand_rotation: bool = True,
        rand_translation: Optional[float] = None,
        data_scale: float = 1,
        num_entities: int = 50,
        scale: float = 1,
        shift: float = 0,
    ):
        self.dataset = MD17Traj(
            root=root,
            molecule_name=molecule_name,
            with_h=with_h,
            down_sample_every=down_sample_every,
            span=span,
            charge_power=charge_power,
            force_reprocess=force_reprocess,
            force_length=force_length,
            mode=mode,
            return_index=return_index,
            project=project,
        )
        self.first_stage = first_stage
        self.rand_rotation = rand_rotation
        self.rand_translation = rand_translation
        self.span = span
        self.data_scale = data_scale
        self.num_entities = num_entities
        self.scale = torch.tensor(scale)
        self.shift = torch.tensor(shift)
        self.cond_index = torch.tensor([dataset_cond_indices[molecule_name]], dtype=torch.long)

    def augment_data(self, pos):
        pos = (pos - self.shift) / self.scale
        if self.rand_rotation:
            rotation_matrix = random_rotation_matrix()
            pos = rotate_point_cloud(pos, R=rotation_matrix)
        if self.rand_translation is not None:
            translation_vector = torch.randn(3) * self.rand_translation
            pos = pos + translation_vector
        return pos

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        pos = item["x"]
        atom = item["z"].long()

        entities = torch.randperm(self.num_entities)[: atom.shape[0]].long()

        # Center the positions around the first frame. Dims here are (A, P, T)
        pos = pos - pos[..., 0].mean(dim=0)[None, ..., None]
        pos = rearrange(pos, "A P T -> (T A) P")
        pos = self.augment_data(pos)
        pos = rearrange(pos, "(T A) P -> T A P", T=self.span)

        if self.first_stage:
            rand_idx = torch.randint(0, pos.shape[0], (1,)).item()
            pos = pos[rand_idx]
        else:
            pos = pos[None]
            atom = atom.repeat(self.span, 1)[None]
            entities = entities.expand(self.span, -1)[None]

        return {
            "pos": pos,
            "atom": atom,
            "cond_molecule": self.cond_index,
            "entities": entities,
        }


class MD17DataModule(L.LightningDataModule):

    def __init__(
        self,
        first_stage: bool,
        molecule_names: list[str],
        root: str,
        with_h: bool,
        down_sample_every: int,
        span: int,
        charge_power: int,
        force_reprocess: bool,
        force_length: Optional[int],
        return_index: bool,
        project: bool,
        rand_rotation: bool,
        rand_translation: Optional[float],
        data_scale: float,
        scale: float,
        shift: float,
        num_entities: int,
        batch_size: int,
        drop_last: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int,
    ):
        super().__init__()
        self.first_stage = first_stage
        self.molecule_names = molecule_names
        self.root = root
        self.with_h = with_h
        self.down_sample_every = down_sample_every
        self.span = span
        self.charge_power = charge_power
        self.force_reprocess = force_reprocess
        self.force_length = force_length
        self.return_index = return_index
        self.project = project
        self.rand_rotation = rand_rotation
        self.rand_translation = rand_translation
        self.data_scale = data_scale
        self.shift = shift
        self.scale = scale
        self.num_entities = num_entities
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.collate_fn = CollatePadBatch() if self.first_stage else CollatePadBatchTemp()

    def dataloader_names(self, idx: int):
        return self.molecule_names[idx]

    def _create_dataloader(self, mode: str, molecule_names: list[str]):
        return DataLoader(
            ConcatDataset(
                [
                    MD17Dataset(
                        first_stage=self.first_stage,
                        molecule_name=molecule_name,
                        root=self.root,
                        with_h=self.with_h,
                        down_sample_every=self.down_sample_every,
                        span=self.span,
                        charge_power=self.charge_power,
                        force_reprocess=self.force_reprocess,
                        force_length=self.force_length,
                        mode=mode,
                        return_index=self.return_index,
                        project=self.project,
                        rand_rotation=self.rand_rotation if mode == "train" else False,
                        rand_translation=self.rand_translation if mode == "train" else 0.0,
                        data_scale=self.data_scale,
                        scale=self.scale,
                        shift=self.shift,
                        num_entities=self.num_entities,
                    )
                    for molecule_name in molecule_names
                ]
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
        """For training, we use all molecules."""
        return self._create_dataloader(mode="train", molecule_names=self.molecule_names)

    def val_dataloader(self):
        """For validation, we use each molecule separately."""
        return [
            self._create_dataloader(mode="val", molecule_names=[molecule_name])
            for molecule_name in self.molecule_names
        ]

    def test_dataloader(self):
        """For testing, we use each molecule separately."""
        return [
            self._create_dataloader(mode="test", molecule_names=[molecule_name])
            for molecule_name in self.molecule_names
        ]

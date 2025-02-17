import lightning as L
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import ConcatDataset

from src.datasets.collate_functions import CollatePadBatch, CollatePadBatchTemp
from src.datasets.geo_tdm.eth_new import ETHNew
from src.utils.data_utils import random_rotation_matrix_2D, rotate_point_cloud

# This have to stay ordered, that conditioning index stays consistent,
# although if we do not train on all molecules, we can just ignore the ones
# that are not used.
dataset_cond_indices: dict[str, int] = {
    "zara1": 0,
    "zara2": 1,
    "univ": 2,
    "hotel": 3,
    "eth": 4,
}


class PedestrianDataset(Dataset):
    """Dataset for Pedestrian.

    This is a wrapper around the implementation from
    https://github.com/hanjq17/GeoTDM/tree/main.
    """

    def __init__(
        self,
        first_stage: bool,
        dataset: str,
        past_frames: int,
        future_frames: int,
        traj_scale: float,
        phase: str,
        return_index: bool = False,
        rand_rotation: bool = False,
        rand_translation: float = None,
        flip_vertical: bool = False,
        flip_horizontal: bool = False,
        num_entities: int = 50,
        shift: float = 0,
        scale: float = 1,
    ):
        self.dataset = ETHNew(
            dataset=dataset,
            past_frames=past_frames,
            future_frames=future_frames,
            traj_scale=traj_scale,
            phase=phase,
            return_index=return_index,
        )

        self.rand_rotation = rand_rotation
        self.rand_translation = rand_translation
        self.first_stage = first_stage
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.shift = shift
        self.scale = scale
        self.num_entities = num_entities
        self.cond_index = torch.tensor([dataset_cond_indices[dataset]], dtype=torch.long)

    def augment_data(
        self,
        pos: torch.Tensor,
    ):
        if self.rand_rotation:
            rotation_matrix = random_rotation_matrix_2D()
            pos = rotate_point_cloud(pos, rotation_matrix)
        if self.flip_vertical:
            pos[:, 0] *= -1
        if self.flip_horizontal:
            pos[:, 1] *= -1
        if self.rand_translation is not None:
            translation_vector = torch.randn(2) * self.rand_translation
            pos = pos + translation_vector
        return pos

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        pos = item["x"]

        N, _, T = pos.size()
        pos = (pos - self.shift) / self.scale
        pos = rearrange(pos, "N D T -> (T N) D")
        pos = self.augment_data(pos)
        pos = rearrange(pos, "(T N) D -> T N D", T=T)

        entities = torch.randperm(self.num_entities)[:N].long()

        if self.first_stage:
            rand_idx = torch.randint(0, N, (1,)).item()
            pos = pos[rand_idx]
        else:
            pos = pos[None]
            entities = entities.expand(T, -1)[None]

        return {
            "pos": pos,
            "cond_scene": self.cond_index,
            "entities": entities,
        }


class PedestrianDataModule(L.LightningDataModule):

    def __init__(
        self,
        first_stage: bool,
        scenes: list[str],
        past_frames: int,
        future_frames: int,
        traj_scale: float,
        return_index: bool,
        rand_rotation: bool,
        rand_translation: float,
        flip_vertical: bool,
        flip_horizontal: bool,
        shift: float,
        scale: float,
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
        self.scenes = scenes
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.traj_scale = traj_scale
        self.return_index = return_index
        self.rand_rotation = rand_rotation
        self.rand_translation = rand_translation
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
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
        return self.scenes[idx]

    def _create_dataloader(self, phase: str, scenes: list[str]):
        return DataLoader(
            ConcatDataset(
                [
                    PedestrianDataset(
                        first_stage=self.first_stage,
                        dataset=scene,
                        past_frames=self.past_frames,
                        future_frames=self.future_frames,
                        traj_scale=self.traj_scale,
                        phase=phase,
                        return_index=self.return_index,
                        rand_rotation=self.rand_rotation if phase == "training" else False,
                        rand_translation=self.rand_translation if phase == "training" else 0.0,
                        flip_vertical=self.flip_vertical if phase == "training" else False,
                        flip_horizontal=self.flip_horizontal if phase == "training" else False,
                        shift=self.shift,
                        scale=self.scale,
                        num_entities=self.num_entities,
                    )
                    for scene in scenes
                ]
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=phase == "training" and self.drop_last,
            shuffle=phase == "training",
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        """For training, we use all scenes."""
        return self._create_dataloader("training", self.scenes)

    def val_dataloader(self):
        """For validation, we use each scene separately.

        Unfortunately this setup does not provide a validation set. Compared methods take the best
        model on the test set, which is in general not recommended, but to stay comparable with
        other methods we use the same setup.
        """
        return [self._create_dataloader("testing", [scene]) for scene in self.scenes]

    def test_dataloader(self):
        """For testing, we use each scene separately."""
        return [self._create_dataloader("testing", [scene]) for scene in self.scenes]

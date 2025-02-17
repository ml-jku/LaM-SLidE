from typing import Dict

import torch
from torch import Tensor
from tqdm import tqdm


def unbatch_predictions(batch_predictions):
    preds = []
    for batch in tqdm(batch_predictions):
        for i in range(len(batch["pos"])):
            n_atoms = batch["n_atoms"][i]
            pred_pos = batch["pos"][i][:n_atoms]
            true_pos = batch["true_pos"][i][:n_atoms]
            atoms = batch["atoms"][i][:n_atoms]
            pred = {"pos": pred_pos, "atoms": atoms, "true_pos": true_pos}
            preds.append(pred)
    return preds


def pad_t_like_x(t: Tensor, x: Tensor) -> Tensor:
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


def batch_to_device(batch: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

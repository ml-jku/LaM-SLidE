from typing import Dict, List, Union

import torch
from einops import rearrange
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def pad_2d(sequences, padding_value=0):
    # First pad the width dimension if needed
    max_width = max(seq.size(1) for seq in sequences)
    sequences = [
        F.pad(seq, (0, max_width - seq.size(1)), value=padding_value) for seq in sequences
    ]
    # Then pad the length dimension
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


class CollatePadBatch:

    def __call__(
        self, batch
    ) -> Dict[str, Union[torch.Tensor, List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]]:
        keys = batch[0].keys()
        padded_batch = {}

        for key in keys:
            sequences = [item[key] for item in batch]
            if key == "adj_matrix":
                padded_batch[key] = pad_2d(sequences)
                continue

            if isinstance(sequences[0], torch.Tensor):
                padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
                padded_batch[key] = padded_sequences
                if "attention_mask" not in padded_batch.keys():
                    attention_mask = (padded_sequences != 0).long()

                    padded_batch["attention_mask"] = attention_mask[:, :, 0].to(dtype=torch.bool)
                continue

            padded_batch[key] = sequences
        return padded_batch


class CollatePadBatchTemp:

    def __call__(
        self, batch
    ) -> Dict[str, Union[torch.Tensor, List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]]:
        B = len(batch)
        keys = batch[0].keys()
        padded_batch = {}

        for key in keys:
            sequences = []
            if isinstance(batch[0][key], torch.Tensor):
                if "cond" in key:
                    padded_batch[key] = torch.cat([item[key] for item in batch], dim=0)
                    continue

                sequences = [
                    seq
                    for item in batch
                    for time_step in item[key].unbind(dim=0)
                    for seq in time_step.unbind(dim=0)
                ]

                padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
                padded_batch[key] = rearrange(padded_sequences, "(B T) S ... -> B T S ...", B=B)
                if "attention_mask" not in padded_batch.keys():
                    attention_mask = (padded_sequences != 0).long()

                    padded_batch["attention_mask"] = attention_mask[:, :, 0].to(dtype=torch.bool)
                    padded_batch["attention_mask"] = rearrange(
                        padded_batch["attention_mask"], "(B T) ... -> B T ...", B=B
                    )
                continue

            sequences = [item[key] for item in batch]
            padded_batch[key] = sequences
        return padded_batch


class CollatePadBatchTempV2:

    def __call__(
        self, batch
    ) -> Dict[str, Union[torch.Tensor, List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]]:
        B = len(batch)
        keys = batch[0].keys()
        padded_batch = {}

        for key in keys:
            sequences = []
            if isinstance(batch[0][key], torch.Tensor):
                if "cond" in key:
                    padded_batch[key] = torch.cat([item[key] for item in batch], dim=0)
                    continue

                sequences = [time_step for item in batch for time_step in item[key].unbind(dim=0)]

                padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
                padded_batch[key] = rearrange(padded_sequences, "(B T) S ... -> B T S ...", B=B)
                if "attention_mask" not in padded_batch.keys():
                    attention_mask = (padded_sequences != 0).long()

                    padded_batch["attention_mask"] = attention_mask[:, :, 0].to(dtype=torch.bool)
                    padded_batch["attention_mask"] = rearrange(
                        padded_batch["attention_mask"], "(B T) ... -> B T ...", B=B
                    )
                continue

            sequences = [item[key] for item in batch]
            padded_batch[key] = sequences
        return padded_batch

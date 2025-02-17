import torch.nn as nn


class TruncNormal:
    def __init__(self, std=0.02) -> None:
        self.std = std

    def __call__(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

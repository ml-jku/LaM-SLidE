import torch
from torch import Tensor, nn


class MaskedMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss: Tensor = self.mse_loss(input, target)
        masked_loss: Tensor = loss.mean(dim=1) * mask
        return masked_loss.sum() / mask.sum()


class MaskedHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=delta)

    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss: Tensor = self.huber_loss(input, target)
        masked_loss: Tensor = loss.mean(dim=1) * mask
        return masked_loss.sum() / mask.sum()


class MaskedNormLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        norm = torch.norm(input - target, dim=-1)
        masked_loss: Tensor = norm * mask
        return masked_loss.sum() / mask.sum()


class MaskedHuberInterDistanceLoss(nn.Module):
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=delta)

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        diag_att: Tensor = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        dist_preds: Tensor = torch.cdist(preds, preds)
        dist_targets: Tensor = torch.cdist(targets, targets)
        loss: Tensor = self.huber_loss(dist_preds, dist_targets) * diag_att
        return loss.sum() / diag_att.sum()


class MaskedL1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.abs_loss = nn.L1Loss(reduction="none")

    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss: Tensor = self.abs_loss(input, target)
        masked_loss: Tensor = loss.mean(dim=1) * mask
        return masked_loss.sum() / mask.sum()


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=label_smoothing
        )

    def forward(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss: Tensor = self.cross_entropy_loss(logits, target)
        masked_loss: Tensor = loss * mask
        return masked_loss.sum() / mask.sum()


class MaskedCosineLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        loss: Tensor = 1 - nn.functional.cosine_similarity(preds, targets, dim=-1)
        masked_loss: Tensor = loss * mask
        return masked_loss.sum() / mask.sum()


class MaskedCosineLossV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        loss: Tensor = 1 - (preds * targets).sum(dim=-1)
        masked_loss: Tensor = loss * mask
        return masked_loss.sum() / mask.sum()


class MaskedCosineLossV3(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        # Calculate difference and clamp
        diff = (preds - targets).abs()
        diff.clamp_min_(1e-3)

        # Square the difference (MSE)
        loss = (diff**2).sum(dim=-1)

        # Apply mask
        masked_loss: Tensor = loss * mask
        return masked_loss.sum() / mask.sum()


class SimilarityLoss(nn.Module):
    def __init__(self, sigma: float = 0.01) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        diag_att: Tensor = torch.triu((mask.unsqueeze(-1) * mask.unsqueeze(-2)), diagonal=1)
        dist: Tensor = torch.cdist(input, input)

        similarity: Tensor = torch.exp(-((dist) ** 2) / (2 * self.sigma**2))
        similarity = torch.triu(similarity, diagonal=1) * diag_att
        return similarity.sum() / mask.sum()


class InterDistanceLoss(nn.Module):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        diag_att: Tensor = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        dist_preds: Tensor = torch.cdist(preds, preds)
        dist_targets: Tensor = torch.cdist(targets, targets)
        dist: Tensor = (dist_preds - dist_targets) * diag_att
        loss: Tensor = dist**2
        return loss.sum() / diag_att.sum()


class InterDistanceLoss2(nn.Module):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        diag_att: Tensor = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        dist_preds: Tensor = torch.cdist(preds, preds)
        dist_targets: Tensor = torch.cdist(targets, targets)
        dist: Tensor = (dist_preds - dist_targets) * diag_att
        return dist.sum() / diag_att.sum()


class InterDistanceLossAdjacent(nn.Module):
    def forward(self, preds: Tensor, targets: Tensor, adj_matrix: Tensor) -> Tensor:
        dist_preds: Tensor = torch.cdist(preds, preds)
        dist_targets: Tensor = torch.cdist(targets, targets)
        dist: Tensor = (dist_preds - dist_targets) * adj_matrix
        loss: Tensor = dist**2
        return loss.sum() / adj_matrix.sum()


class InterDistanceLossV2(nn.Module):
    def __init__(self, relative: bool = True):
        super().__init__()
        self.relative = relative

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        diag_att: Tensor = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        dist_preds: Tensor = torch.cdist(preds, preds)
        dist_targets: Tensor = torch.cdist(targets, targets)

        if self.relative:
            # Relative differences might be more meaningful for molecular structures
            dist = torch.abs(dist_preds - dist_targets) / (dist_targets + 1e-8)
        else:
            # Absolute differences
            dist = torch.abs(dist_preds - dist_targets)

        loss: Tensor = dist * diag_att
        return loss.sum() / diag_att.sum()

import math

from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup and cosine annealing scheduler.

    Has to be used in combination with `src.callbacks.step_lr.StepLRCallback`.

    Modifies the learning rate every gradient step, but is configured based on epochs.
    """

    def __init__(
        self, optimizer, warmup_epochs, max_epochs, steps_per_epoch=1, min_lr=0.0, last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.set_steps_per_epoch(steps_per_epoch)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.max_steps = self.max_epochs * self.steps_per_epoch

    def get_lr(self):
        current_step = self.last_epoch + 1

        if current_step <= self.warmup_steps:
            # Linear warmup
            return [base_lr * current_step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs
            ]
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * 0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.warmup_epochs)
                        / (self.total_epochs - self.warmup_epochs)
                    )
                )
                for base_lr in self.base_lrs
            ]

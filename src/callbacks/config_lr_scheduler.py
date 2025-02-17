import lightning as L
from lightning import Callback


class ConfigLRScheduler(Callback):
    """Count up every gradient update step rather than every epoch."""

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if len(trainer.lr_scheduler_configs) > 0:
            self.scheduler = trainer.lr_scheduler_configs[0].scheduler
            assert self.scheduler.__class__.__name__ == "LinearWarmupCosineAnnealingLR"
            self.scheduler.set_steps_per_epoch(
                len(trainer.train_dataloader) // trainer.accumulate_grad_batches
            )

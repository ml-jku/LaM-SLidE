from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
from einops import rearrange
from hydra.utils import instantiate
from torch import Tensor, nn

from src.modules.transport.transport import Sampler, Transport
from src.utils.pylogger import RankedLogger
from src.utils.tensor_utils import tensor_tree_map

log = RankedLogger(__name__, rank_zero_only=True)


class BackboneBase(ABC, nn.Module):
    def __init__(self, dim_latent: int, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.dim_latent = dim_latent
        self.encoder = encoder
        self.decoder = decoder

        self.quant = nn.Sequential(
            nn.Linear(dim_latent, dim_latent),
            nn.LayerNorm(dim_latent, elementwise_affine=False),
        )
        self.post_quant = nn.Sequential(
            nn.LayerNorm(dim_latent, elementwise_affine=False),
            nn.Linear(dim_latent, dim_latent),
        )

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        z = self.encode(batch)
        return self.decode(z=z, entities=batch["entities"])

    def encode(self, batch: Dict[str, Tensor]) -> Tensor:
        x = self.prepare_inputs(batch)
        latents = self.encoder(x=x, entities=batch["entities"], mask=batch["attention_mask"])
        return self.quant(latents)

    def decode(self, z: Tensor, entities: Tensor) -> Tensor:
        latents = self.post_quant(z)
        return self.decoder(latents, entities)

    @abstractmethod
    def prepare_inputs(self, batch: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError


class LightningBase(L.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.ema = None
        self.cached_weights = None

    def setup(self, stage: Optional[str] = None):
        if self.hparams.ema is not None:
            self.ema = instantiate(self.hparams.ema)(model=self)

    def load_ema_weights(self):
        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling
        # load_state_dict().
        print("Loading EMA weights")
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.state_dict())
        self.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        print("Restoring cached weights")
        if self.cached_weights is not None:
            self.load_state_dict(self.cached_weights)
            self.cached_weights = None

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema:
            self.ema.update(self)

    def on_train_start(self):
        if self.ema:
            if self.ema.device != self.device:
                self.ema.to(self.device)

    def on_validation_start(self):
        if self.ema:
            if self.ema.device != self.device:
                self.ema.to(self.device)
            if self.cached_weights is None:
                self.load_ema_weights()

    def on_validation_end(self):
        if self.ema:
            self.restore_cached_weights()

    def on_test_start(self):
        if self.ema:
            if self.ema.device != self.device:
                self.ema.to(self.device)
            if self.cached_weights is None:
                self.load_ema_weights()

    def on_test_end(self):
        if self.ema:
            self.restore_cached_weights()

    def on_load_checkpoint(self, checkpoint):
        print(f"Loading EMA state dict from checkpoint {checkpoint['epoch']}")
        if self.hparams.ema is not None:
            self.ema = instantiate(self.hparams.ema)(model=self)
            self.ema.load_state_dict(checkpoint["ema"])

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            if self.cached_weights is not None:
                self.restore_cached_weights()
            checkpoint["ema"] = self.ema.state_dict()

    def configure_optimizers(self) -> Dict[str, Any]:
        self.lr = self.hparams.optimizer.lr
        optimizer = instantiate(self.hparams.optimizer)(
            params=filter(lambda p: p.requires_grad, self.trainer.model.parameters())
        )
        if self.hparams.scheduler is not None:
            scheduler = instantiate(self.hparams.scheduler)(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor,
                    "interval": self.hparams.interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class FirstStageLightningBase(LightningBase):

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.backbone(batch)

    def model_step(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        loss, preds = self.loss(model=self, batch=batch)
        return loss, preds

    def decode(self, latents: Tensor, entities: Tensor) -> torch.Tensor:
        return self.backbone.decode(
            z=latents,
            entities=entities,
        )

    def encode(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.backbone.encode(batch)

    @property
    def shift(self) -> float:
        return self.hparams.shift

    @property
    def scale(self) -> float:
        return self.hparams.scale


class SecondStageCondLightningBase(LightningBase, ABC):
    """Base class for second stage models that use conditioning.

    This class depends on hparams set in subclass, not clean but convenient for development.
    """

    def forward(self, xt: Tensor, t: Tensor, **model_kwargs) -> Tensor:
        return self.backbone(x=xt, t=t, **model_kwargs)

    @property
    def scale(self) -> float:
        return self.first_stage_model.hparams.scale

    @property
    def shift(self) -> float:
        return self.first_stage_model.hparams.shift

    @property
    def n_timesteps(self) -> int:
        return self.hparams.n_timesteps

    @property
    def transport(self) -> Transport:
        return self.si

    def model_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = self.prepare_batch(batch)
        loss, preds = self.loss(model=self, batch=batch)
        return loss, preds

    @abstractmethod
    def encode(self, batch: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latents: Tensor, entities: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def prepare_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        latents = self.encode(batch)
        x_cond, x_cond_mask = self.setup_conditioning(latents)
        model_kwargs = {
            "x_cond": x_cond,
            "x_cond_mask": x_cond_mask,
        }
        batch["x1"] = latents
        batch["model_kwargs"] = model_kwargs
        return batch

    @torch.no_grad()
    def sample(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_fn = Sampler(self.si).get_sample_fn(
            self.hparams.sampling_method, self.hparams.sampling_kwargs
        )
        B = batch["entities"].size()[0]

        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor) and batch[key].device != self.device:
                batch[key] = batch[key].to(self.device)

        batch = self.prepare_batch(batch)
        model_kwargs = batch["model_kwargs"]
        latents = sample_fn(
            torch.randn_like(model_kwargs["x_cond"], device=self.device),
            self.forward,
            **model_kwargs,
        )[-1]

        latents = rearrange(latents, "B T ... -> (B T) ...", B=B)
        entities = rearrange(batch["entities"], "B T ... -> (B T) ...", B=B)
        return self.decode(latents, entities)

    @torch.no_grad()
    def setup_conditioning(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        """Setup conditioning for the second stage model.

        If mask_cond_mean is True, the conditioning is set to the mean of the visible/conditioning
        latents. Otherwise zero, we observe slightly better convergence with mean conditioning.
        """
        B, T, L, _ = latents.size()
        x_cond_mask = torch.zeros(B, T, L, dtype=int, device=self.device)
        x_cond_mask[:, self.hparams.cond_idx[0] : self.hparams.cond_idx[1]] = 1

        if self.hparams.mask_cond_mean:
            # Empirically, using the mean of the conditioning latents leads to slightly
            # better convergence.
            x_cond = torch.where(
                x_cond_mask.unsqueeze(-1).bool(),
                latents,
                latents[:, self.hparams.cond_idx[0] : self.hparams.cond_idx[1]]
                .mean(dim=1)
                .unsqueeze(1),
            )
        else:
            x_cond = torch.where(x_cond_mask.unsqueeze(-1).bool(), latents, 0.0)
        return x_cond, x_cond_mask

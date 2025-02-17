import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.models.components.latent.mmdit import (
    EmbedND,
    MLPEmbedder,
    Modulation,
    ParallelMLPAttentionV2,
    modulate,
    timestep_embedding,
)


class LatentSIV3Layer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        attention_mode: str = "scaled_dot_product",
    ):
        super().__init__()
        self.modulation = Modulation(hidden_size, double=True)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.spatial_block = ParallelMLPAttentionV2(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_mode=attention_mode,
        )
        self.temporal_block = ParallelMLPAttentionV2(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_mode=attention_mode,
        )

    def forward(self, x: Tensor, y: Tensor, pe_spatial: EmbedND, pe_temporal: EmbedND) -> Tensor:
        _, T, L, _ = x.size()

        mod1, mod2 = self.modulation(y)
        residual = x
        x = modulate(self.pre_norm(x), mod1.shift, mod1.scale)
        x = rearrange(x, "B T L D -> (B T) L D", L=L)
        x = self.spatial_block(x=x, pe=pe_spatial)
        x = rearrange(x, "(B T) L D -> B T L D", T=T)
        x = residual + mod1.gate.unsqueeze(1) * x

        residual = x
        x = modulate(self.pre_norm(x), mod2.shift, mod2.scale)
        x = rearrange(x, "B T L D -> (B L) T D", L=L)
        x = self.temporal_block(x=x, pe=pe_temporal)
        x = rearrange(x, "(B L) T D -> B T L D", L=L)
        x = residual + mod2.gate.unsqueeze(1) * x

        return x


class LatentSIV3(nn.Module):

    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_size: int,
        num_heads: int,
        vec_in_dim: Optional[int] = None,
        mlp_ratio: int = 2,
        n_timesteps: int = 10,
        theta: int = 10_000,
        checkpointing: bool = False,
        normalize: bool = False,
        attention_mode: str = "scaled_dot_product",
        share_weights: bool = False,
        reset_parameters: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.n_timesteps = n_timesteps
        self.checkpointing = checkpointing
        self.normalize = normalize
        self.attention_mode = attention_mode

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads

        self.x_in = nn.Linear(in_dim, hidden_size)
        self.cond_to_emb = nn.Linear(in_dim, hidden_size)
        self.mask_to_emb = nn.Embedding(2, hidden_size)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
        if vec_in_dim is not None:
            self.vec_in = MLPEmbedder(in_dim=vec_in_dim, hidden_dim=hidden_size)
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=[pe_dim])
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.blocks = nn.ModuleList()
        if share_weights:
            block = LatentSIV3Layer(hidden_size, num_heads, mlp_ratio, attention_mode)
            for _ in range(depth):
                self.blocks.append(block)
        else:
            for _ in range(depth):
                self.blocks.append(
                    LatentSIV3Layer(hidden_size, num_heads, mlp_ratio, attention_mode)
                )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, self.out_dim)

        if reset_parameters:
            self.reset_parameters()

    def reset_parameters(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.time_in.in_layer.weight, std=0.02)
        nn.init.normal_(self.time_in.out_layer.weight, std=0.02)

        if hasattr(self, "vec_in"):
            nn.init.normal_(self.vec_in.in_layer.weight, std=0.02)
            nn.init.normal_(self.vec_in.out_layer.weight, std=0.02)

        for block in self.blocks:
            nn.init.xavier_uniform_(block.spatial_block.linear1.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(block.spatial_block.linear2.weight, gain=1 / math.sqrt(2))
            nn.init.constant_(block.spatial_block.linear2.bias, 0.0)

            nn.init.xavier_uniform_(block.temporal_block.linear1.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(block.temporal_block.linear2.weight, gain=1 / math.sqrt(2))
            nn.init.constant_(block.temporal_block.linear2.bias, 0.0)

            nn.init.constant_(block.modulation.lin.weight, 0.0)
            nn.init.constant_(block.modulation.lin.bias, 0.0)

        nn.init.normal_(self.linear.weight, std=0.00)
        nn.init.normal_(self.linear.bias, std=0.00)

    def temporal_rope_embedding(self, B: int, T: int, L: int, device: torch.device) -> Tensor:
        return self.pe_embedder(torch.arange(T, device=device)[None, :, None]).expand(
            B * L, -1, -1, -1, -1, -1
        )

    def spatial_rope_embedding(self, B: int, T: int, L: int, device: torch.device) -> Tensor:
        return self.pe_embedder(torch.arange(L, device=device)[None, :, None]).expand(
            B * T, -1, -1, -1, -1, -1
        )

    def forward(
        self, x: Tensor, t: Tensor, x_cond: Tensor, x_cond_mask: Tensor, y: Tensor = None
    ) -> Tensor:
        B, T, L, _ = x.size()
        x = self.x_in(x) + self.cond_to_emb(x_cond) + self.mask_to_emb(x_cond_mask)
        if self.normalize:
            x = nn.functional.layer_norm(x, (x.size(-1),))

        vec = self.time_in(timestep_embedding(t, 256))
        if y is not None:
            vec = vec + self.vec_in(y)

        pe_spatial = self.spatial_rope_embedding(B, T, L, x.device)
        pe_temporal = self.temporal_rope_embedding(B, T, L, x.device)
        for block in self.blocks:
            x = block(x=x, y=vec, pe_spatial=pe_spatial, pe_temporal=pe_temporal)

        shift, scale = self.adaLN_modulation(vec)[:, None, :].chunk(2, dim=-1)
        x = modulate(self.pre_norm(x), shift, scale)
        x = self.linear(x)
        return x

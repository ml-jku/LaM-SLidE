from abc import ABC
from functools import partial

import torch
import torch.nn as nn
from einops import repeat

from src.modules.torch_modules import CrossAttentionBlock, SelfAttentionBlock


class EncoderBase(ABC, nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_latent: int,
        num_latents: int,
        entity_embedding: nn.Module,
        dropout_latent: float = 0.0,
        act: nn.Module = partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__()
        self.entity_embedding = entity_embedding
        self.dropout_latent = nn.Dropout2d(dropout_latent)
        self.dim_input = dim_input
        self.dim_context = dim_input + self.entity_embedding.embedding_dim

        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))

        self.mlp = nn.Sequential(
            nn.Linear(self.dim_context, dim_latent),
            act(),
            nn.Linear(dim_latent, self.dim_context),
        )

    def prepare_inputs(self, x, entities):
        entity_embeddings = self.entity_embedding(entities)
        x = torch.cat([x, entity_embeddings], dim=-1)
        x = self.mlp(x)
        latents = repeat(self.latents, "N D -> B N D", B=x.shape[0])
        latents = self.dropout_latent(latents)
        return x, latents


class Encoder(EncoderBase):

    def __init__(
        self,
        dim_input: int,
        dim_latent: int,
        dim_head_cross: int,
        dim_head_latent: int,
        num_latents: int,
        num_head_cross: int,
        num_head_latent: int,
        num_block_cross: int,
        num_block_attn: int,
        qk_norm: bool,
        entity_embedding: nn.Module,
        dropout_latent: float = 0.0,
        act: nn.Module = partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__(
            dim_input=dim_input,
            dim_latent=dim_latent,
            num_latents=num_latents,
            entity_embedding=entity_embedding,
            act=act,
            dropout_latent=dropout_latent,
        )
        self.cross_attn_blocks = nn.ModuleList([])

        for _ in range(num_block_cross):
            self.cross_attn_blocks.append(
                CrossAttentionBlock(
                    dim=dim_latent,
                    heads=num_head_cross,
                    dim_head=dim_head_cross,
                    act=act,
                    context_dim=self.dim_context,
                    qk_norm=qk_norm,
                )
            )

        self.blocks_attn = nn.ModuleList([])
        for _ in range(num_block_attn):
            self.blocks_attn.append(
                SelfAttentionBlock(
                    dim=dim_latent,
                    heads=num_head_latent,
                    dim_head=dim_head_latent,
                    act=act,
                    qk_norm=qk_norm,
                )
            )

    def forward(self, x, entities, mask=None):
        x, latents = self.prepare_inputs(x, entities)

        for cross_attn in self.cross_attn_blocks:
            latents = cross_attn(latents, context=x, mask=mask)
        for self_attn in self.blocks_attn:
            latents = self_attn(latents)
        return latents


class Encoder2(EncoderBase):

    def __init__(
        self,
        dim_input: int,
        dim_latent: int,
        dim_head_cross: int,
        dim_head_latent: int,
        num_latents: int,
        num_head_cross: int,
        num_head_latent: int,
        num_block: int,
        qk_norm: bool,
        entity_embedding: nn.Module,
        dropout_latent: float = 0.0,
        act: nn.Module = partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__(
            dim_input=dim_input,
            dim_latent=dim_latent,
            num_latents=num_latents,
            entity_embedding=entity_embedding,
            act=act,
            dropout_latent=dropout_latent,
        )
        self.cross_attn_blocks = nn.ModuleList([])

        for _ in range(num_block):
            self.cross_attn_blocks.append(
                nn.ModuleList(
                    [
                        CrossAttentionBlock(
                            dim=dim_latent,
                            heads=num_head_cross,
                            dim_head=dim_head_cross,
                            act=act,
                            context_dim=self.dim_context,
                            qk_norm=qk_norm,
                        ),
                        SelfAttentionBlock(
                            dim=dim_latent,
                            heads=num_head_latent,
                            dim_head=dim_head_latent,
                            act=act,
                            qk_norm=qk_norm,
                        ),
                    ]
                )
            )

    def forward(self, x, entities, mask=None):
        x, latents = self.prepare_inputs(x, entities)

        for cross_attn, self_attn in self.cross_attn_blocks:
            latents = cross_attn(latents, context=x, mask=mask)
            latents = self_attn(latents)
        return latents

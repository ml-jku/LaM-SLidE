from functools import partial
from typing import Dict

import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from src.modules.torch_modules import CrossAttentionBlock, SelfAttentionBlock


class Decoder(nn.Module):

    def __init__(
        self,
        outputs: Dict[str, int],
        dim_query: int,
        dim_latent: int,
        entity_embedding: nn.Module,
        dim_head_cross: int = 64,
        dim_head_latent: int = 64,
        num_head_cross: int = 1,
        num_head_latent: int = 4,
        num_block_cross: int = 2,
        num_block_attn: int = 4,
        dropout_query: float = 0.1,
        dropout_latent: float = 0.0,
        qk_norm: bool = False,
        act: nn.Module = partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__()

        self.entity_embedding = entity_embedding
        dim_entity_embedding = self.entity_embedding.embedding_dim
        self.query_mlp = nn.Sequential(
            nn.Dropout(dropout_query),
            nn.Linear(dim_entity_embedding, dim_query),
        )
        self.dropout_latent = nn.Dropout(dropout_latent)
        self.self_attn_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_latent,
                    heads=num_head_latent,
                    dim_head=dim_head_latent,
                    act=act,
                    qk_norm=qk_norm,
                )
                for _ in range(num_block_attn)
            ]
        )
        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=dim_latent,
                    heads=num_head_cross,
                    dim_head=dim_head_cross,
                    act=act,
                    context_dim=dim_query,
                    qk_norm=qk_norm,
                )
                for _ in range(num_block_cross)
            ]
        )

        self.output_block = CrossAttentionBlock(
            dim=dim_query,
            heads=num_head_cross,
            dim_head=dim_head_cross,
            act=act,
            context_dim=dim_latent,
            qk_norm=qk_norm,
        )

        self.output_layers = nn.ModuleDict()
        for name, out_dim in outputs.items():
            self.output_layers[name] = nn.Sequential(
                nn.Linear(dim_query, dim_query),
                act(),
                nn.Linear(dim_query, out_dim),
            )

    def queries(self, entities):
        entity_embeddings = self.entity_embedding(entities)
        queries = self.query_mlp(entity_embeddings)
        return queries

    def forward(self, latent, entities):
        queries = self.queries(entities)

        latent = self.dropout_latent(latent)
        for block in self.self_attn_blocks:
            latent = block(latent)
        for block in self.cross_attn_blocks:
            latent = block(latent, context=queries)

        out_block = self.output_block(queries, context=latent)

        outputs = {}
        for name in self.output_layers.keys():
            outputs[name] = self.output_layers[name](out_block)
        return outputs


class DecoderFE(nn.Module):

    def __init__(
        self,
        outputs: Dict[str, int],
        dim_query: int,
        dim_latent: int,
        entity_embedding: nn.Module,
        dim_head_cross: int = 64,
        dim_head_latent: int = 64,
        num_head_cross: int = 1,
        num_head_latent: int = 4,
        num_block_cross: int = 2,
        num_block_attn: int = 4,
        dropout_query: float = 0.1,
        dropout_latent: float = 0.0,
        qk_norm: bool = False,
        act: nn.Module = partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__()

        self.entity_embedding = entity_embedding
        dim_entity_embedding = self.entity_embedding.embedding_dim
        self.query_mlp = nn.Sequential(
            nn.Dropout(dropout_query),
            nn.Linear(dim_entity_embedding, dim_query),
        )
        self.dropout_latent = nn.Dropout(dropout_latent)
        self.self_attn_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_latent,
                    heads=num_head_latent,
                    dim_head=dim_head_latent,
                    act=act,
                    qk_norm=qk_norm,
                )
                for _ in range(num_block_attn)
            ]
        )
        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=dim_latent,
                    heads=num_head_cross,
                    dim_head=dim_head_cross,
                    act=act,
                    context_dim=dim_query,
                    qk_norm=qk_norm,
                )
                for _ in range(num_block_cross)
            ]
        )

        self.output_block = CrossAttentionBlock(
            dim=dim_query,
            heads=num_head_cross,
            dim_head=dim_head_cross,
            act=act,
            context_dim=dim_latent,
            qk_norm=qk_norm,
        )

        self.output_layers = nn.ModuleDict()
        for name, out_dim in outputs.items():
            self.output_layers[name] = nn.Sequential(
                nn.Linear(dim_query, dim_query),
                act(),
                nn.Linear(dim_query, out_dim),
            )

        self.energy_query = nn.Parameter(torch.randn(dim_query))
        self.energy_block = CrossAttentionBlock(
            dim=dim_query,
            heads=num_head_cross,
            dim_head=dim_head_cross,
            act=act,
            context_dim=dim_latent,
            qk_norm=qk_norm,
        )
        self.energy_mlp = nn.Sequential(
            nn.Linear(dim_query, dim_query),
            act(),
            nn.Linear(dim_query, 1),
        )

    def queries(self, entities):
        entity_embeddings = self.entity_embedding(entities)
        queries = self.query_mlp(entity_embeddings)
        return queries

    def forward(self, latent, entities):
        queries = self.queries(entities)

        latent = self.dropout_latent(latent)
        for block in self.self_attn_blocks:
            latent = block(latent)
        for block in self.cross_attn_blocks:
            latent = block(latent, context=queries)

        out_block = self.output_block(queries, context=latent)

        outputs = {}
        for name in self.output_layers.keys():
            outputs[name] = self.output_layers[name](out_block)

        energy_queries = repeat(self.energy_query, "D -> B 1 D", B=entities.shape[0])
        energy_block = self.energy_block(energy_queries, context=latent)
        energy_mlp = self.energy_mlp(energy_block)
        outputs["energy"] = energy_mlp.squeeze(-1)

        return outputs


class Decoder2(nn.Module):

    def __init__(
        self,
        outputs: Dict[str, int],
        dim_query: int,
        dim_latent: int,
        entity_embedding: nn.Module,
        dim_head_cross: int = 64,
        dim_head_latent: int = 64,
        num_head_cross: int = 1,
        num_head_latent: int = 4,
        num_block_cross: int = 2,
        num_block_attn: int = 4,
        dropout_query: float = 0.1,
        dropout_latent: float = 0.0,
        qk_norm: bool = False,
        act: nn.Module = partial(nn.GELU, approximate="tanh"),
    ):
        super().__init__()
        self.entity_embedding = entity_embedding
        dim_entity_embedding = self.entity_embedding.embedding_dim
        self.query_mlp = nn.Sequential(
            nn.Dropout(dropout_query),
            nn.Linear(dim_entity_embedding, dim_query),
        )
        self.query = nn.Parameter(torch.randn(dim_query))
        self.dropout_latent = nn.Dropout(dropout_latent)
        self.self_attn_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_latent,
                    heads=num_head_latent,
                    dim_head=dim_head_latent,
                    act=act,
                    qk_norm=qk_norm,
                )
                for _ in range(num_block_attn)
            ]
        )
        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=dim_latent,
                    heads=num_head_cross,
                    dim_head=dim_head_cross,
                    act=act,
                    context_dim=dim_query,
                    qk_norm=qk_norm,
                )
                for _ in range(num_block_cross)
            ]
        )

        self.output_block = CrossAttentionBlock(
            dim=dim_query,
            heads=num_head_cross,
            dim_head=dim_head_cross,
            act=act,
            context_dim=dim_latent,
            qk_norm=qk_norm,
        )

        self.output_layers = nn.ModuleDict()
        for name, out_dim in outputs.items():
            self.output_layers[name] = nn.Sequential(
                nn.Linear(dim_query, dim_query),
                act(),
                nn.Linear(dim_query, out_dim),
            )

    def queries(self, entities):
        query = repeat(self.query, "D -> B N D", B=entities.shape[0], N=entities.shape[1])
        entity_embeddings = self.entity_embedding(entities)
        queries = self.query_mlp(entity_embeddings)
        return queries + query

    def forward(self, latent, entities):
        queries = self.queries(entities)

        latent = self.dropout_latent(latent)
        for block in self.self_attn_blocks:
            latent = block(latent)
        for block in self.cross_attn_blocks:
            latent = block(latent, context=queries)

        out_block = self.output_block(queries, context=latent)

        outputs = {}
        for name in self.output_layers.keys():
            outputs[name] = self.output_layers[name](out_block)
        return outputs


class DecoderQuerySplitter(nn.Module):

    def __init__(
        self,
        outputs: Dict[str, int],
        dim_query: int,
        dim_latent: int,
        entity_embedding: nn.Module,
        dim_head_cross: int = 64,
        dim_head_latent: int = 64,
        num_head_cross: int = 1,
        num_head_latent: int = 4,
        num_block_cross: int = 2,
        num_block_attn: int = 4,
        dropout_query: float = 0.1,
        dropout_latent: float = 0.0,
        qk_norm: bool = False,
        act: nn.Module = partial(nn.GELU, approximate="tanh"),
        num_split: int = 8,
    ):
        super().__init__()

        self.entity_embedding = entity_embedding
        dim_entity_embedding = self.entity_embedding.embedding_dim
        self.query_mlp = nn.Sequential(
            nn.Dropout(dropout_query),
            nn.Linear(dim_entity_embedding, dim_query),
        )
        self.dropout_latent = nn.Dropout(dropout_latent)
        self.self_attn_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_latent,
                    heads=num_head_latent,
                    dim_head=dim_head_latent,
                    act=act,
                    qk_norm=qk_norm,
                )
                for _ in range(num_block_attn)
            ]
        )
        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=dim_latent,
                    heads=num_head_cross,
                    dim_head=dim_head_cross,
                    act=act,
                    context_dim=dim_query,
                    qk_norm=qk_norm,
                )
                for _ in range(num_block_cross)
            ]
        )

        self.output_block = CrossAttentionBlock(
            dim=dim_query,
            heads=num_head_cross,
            dim_head=dim_head_cross,
            act=act,
            context_dim=dim_latent,
            qk_norm=qk_norm,
        )

        self.output_layers = nn.ModuleDict()
        for name, out_dim in outputs.items():
            self.output_layers[name] = nn.Sequential(
                nn.Linear(dim_query, dim_query),
                act(),
                nn.Linear(dim_query, out_dim),
            )

        self.extender = nn.Sequential(
            Rearrange("B L D -> B D L"),
            nn.Conv1d(dim_latent, dim_latent * num_split, 1),
            Rearrange("B (D N) L -> B (L N) D", N=num_split),
        )

    def queries(self, entities):
        entity_embeddings = self.entity_embedding(entities)
        queries = self.query_mlp(entity_embeddings)
        return queries

    def forward(self, latent, entities):
        queries = self.queries(entities)

        latent = self.dropout_latent(latent)
        for block in self.self_attn_blocks:
            latent = block(latent)
        for block in self.cross_attn_blocks:
            latent = block(latent, context=queries)

        latent = self.extender(latent)
        out_block = self.output_block(queries, context=latent)

        outputs = {}
        for name in self.output_layers.keys():
            outputs[name] = self.output_layers[name](out_block)
        return outputs

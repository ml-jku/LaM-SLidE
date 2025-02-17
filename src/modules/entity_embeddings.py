from typing import Optional

import torch.nn as nn
import torch.nn.init as init


class EntityEmbeddingOrthogonal(nn.Module):
    n_entity_embeddings: int
    embedding_dim: int
    max_norm: Optional[float] = None
    requires_grad: bool = False

    def __init__(
        self,
        n_entiy_embeddings,
        embedding_dim,
        max_norm: Optional[float] = None,
        requires_grad: bool = False,
    ):
        super().__init__()
        self.n_entity_embeddings = n_entiy_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm

        self.embedding = nn.Embedding(n_entiy_embeddings, embedding_dim, max_norm=max_norm)
        init.orthogonal_(self.embedding.weight)
        self.embedding.weight.requires_grad = requires_grad

    def forward(self, entities):
        return self.embedding(entities)

import numpy as np
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    if num_encodings == 0:
        return x
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


class SinCosPositionalEmbedding1D(nn.Module):
    def __init__(self, n_positions, embed_dim):
        super().__init__()
        emb = get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(n_positions))
        self.register_buffer("embeddings", torch.from_numpy(emb).float())  # (num_positions, D)

    def forward(self, x):
        n_res = x.size(1)
        return x + self.embeddings[:n_res][None]


class PointEmbed(nn.Module):
    embedding_dim: int
    hidden_dim: int

    def __init__(self, hidden_dim=48, embedding_dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        e = torch.pow(2, torch.arange(self.hidden_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [e, torch.zeros(self.hidden_dim // 6), torch.zeros(self.hidden_dim // 6)]
                ),
                torch.cat(
                    [torch.zeros(self.hidden_dim // 6), e, torch.zeros(self.hidden_dim // 6)]
                ),
                torch.cat(
                    [torch.zeros(self.hidden_dim // 6), torch.zeros(self.hidden_dim // 6), e]
                ),
            ]
        )
        self.register_buffer("basis", e)  # 3 x 16
        self.mlp = nn.Linear(self.hidden_dim + 3, self.embedding_dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))  # B x N x C
        return embed


def mask_emb(x, p=0.1, train=True):
    """Applies dropout to entire rows along the S dimension of a tensor without scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (B, S, D).
        p (float): Probability of dropping a row. Default is 0.5.
        training (bool): Apply dropout if True. Default is True.

    Returns:
        torch.Tensor: Tensor with rows randomly zeroed out.
    """
    if not train or p == 0:
        return x
    B, S, _ = x.shape
    mask = x.new_empty(B, S, 1).bernoulli_(1 - p)
    return x * mask

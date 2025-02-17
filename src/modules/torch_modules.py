import math
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class GELU(nn.Module):

    @staticmethod
    def gelu(x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        """

        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        return self.gelu(x)


class Sin(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(input)


def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device=device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(seq_keep_counts, "b -> b 1")

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: torch.Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, context=None, mask=None):
        x = self.norm(x)

        if exists(self.norm_context):
            context = self.norm_context(context)
            return self.fn(x, context, mask)

        return self.fn(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        act: nn.Module = nn.GELU,
        input_dim: int = None,
        output_dim: int = None,
    ):
        super().__init__()
        input_dim = default(input_dim, dim)
        output_dim = default(output_dim, dim)
        layers = [nn.Sequential(nn.Linear(input_dim, dim), act())]

        layers = layers + [nn.Sequential(nn.Linear(dim, dim), act()) for _ in range(1, depth)]
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, scale=None, qk_norm=False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = default(scale, dim_head**-0.5)
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.norm = QKNorm(dim_head) if qk_norm else lambda q, k, v: (q, k)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.to_kv.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.constant_(self.to_out.bias, 0)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        q, k = self.norm(q, k, v)

        if mask is not None:
            mask = repeat(mask, "b j -> (b h) () j", h=h)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        context_dim=None,
        heads: int = 4,
        dim_head: int = 64,
        act=nn.GELU,
        scale=None,
        qk_norm=False,
    ):
        super().__init__()
        self.attn = PreNorm(
            dim,
            Attention(
                query_dim=dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=dim_head,
                scale=scale,
                qk_norm=qk_norm,
            ),
            context_dim=context_dim,
        )
        self.ff = PreNorm(dim, FeedForward(dim, act=act))

    def forward(self, x, context=None, mask=None):
        x = self.attn(x, context=context, mask=mask) + x
        x = self.ff(x) + x
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, scale=None, qk_norm=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = default(scale, dim_head**-0.5)
        self.heads = heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.norm = QKNorm(dim_head) if qk_norm else lambda q, k, _: (q, k)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qkv.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.constant_(self.to_out.bias, 0)

    def forward(self, x, mask=None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        q, k = self.norm(q, k, v)

        if mask is not None:
            mask = repeat(mask, "b j -> (b h) () j", h=h)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int = 64,
        act=nn.GELU,
        scale=None,
        qk_norm=False,
    ):
        super().__init__()
        self.attn = PreNorm(dim, SelfAttention(dim, heads, dim_head, scale, qk_norm))
        self.ff = PreNorm(dim, FeedForward(dim, act=act))

    def forward(self, x, mask=None):
        x = self.attn(x, mask=mask) + x
        x = self.ff(x) + x
        return x


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


def grad_checkpoint(func, args, checkpointing=False):
    if checkpointing:
        return checkpoint(func, *args, use_reentrant=False)
    else:
        return func(*args)

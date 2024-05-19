import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import ModelConfig
from .rotary_encoding import RotaryEncoding

class GLU(nn.Module):
    def __init__(self, in_dim, out_dim, activation, bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.activation = activation
        self.w = nn.Linear(in_dim, out_dim * 2, bias=bias)

    def forward(self, x: torch.Tensor):
        w1, w2 = self.w(x).split(self.out_dim)
        return w1 * self.activation(w2)

class BatchLinear(nn.Module):
    def __init__(self, n_streams: int, d_in: int, d_out: int, has_bias = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.has_bias = has_bias

        self.weight = nn.Parameter(torch.zeros((n_streams, d_out, d_in)))
        if has_bias:
            self.bias = nn.Parameter(torch.zeros((n_streams, d_out)))

        self.reset_parameters()

    def reset_parameters(self):
        # uniform initialization between (-1/sqrt(in_features), 1/sqrt(in_features))
        bound = 1 / math.sqrt(self.d_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.has_bias:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        # in: (batch, stream, d_embed)
        # x @ W^T + b
        # transpose to (stream, batch, d_embed) then batch matmul and transpose back
        x = torch.bmm(x.transpose(-2, -3), self.weight.mT).transpose(-2, -3)
        if self.has_bias:
            # add bias (broadcasted): (batch, stream, d_embed) + (1, stream, d_embed)
            x += self.bias

        return x

# TODO: test encoder, needs additional modifications
class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_attention_heads = config.n_attention_heads
        self.d_ff_inner = config.d_ff_inner
        self.activation = config.get_activation()

        self.input_embedding = nn.Embedding(config.vocab_size, config.d_embed, dtype=config.dtype)
        # marker for uncommitted
        # self.uncommited_marker = nn.Parameter(torch.randn((config.d_embed,)))
        # self.norm = nn.LayerNorm((config.d_embed))
        self.init_q = nn.Parameter(
            # (n_heads, stream, n_embed)
            torch.randn((config.n_attention_heads, config.n_streams, config.d_embed))
        )
        self.rope = RotaryEncoding(config.d_embed, config.n_streams)
        self.kv_linear = nn.Linear(
            config.d_embed,
            2 * config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads,)))

        ff_in_dim = config.d_embed * config.n_attention_heads
        self.ff_in = BatchLinear(config.n_streams, ff_in_dim, config.d_ff_inner * 2)
        self.ff_out = BatchLinear(config.n_streams, config.d_ff_inner, config.d_embed)

    def forward(self, x: torch.Tensor):
        # (batch, stream) -> (batch, stream, d_embed)
        embed = self.input_embedding(x)
        # apply rotary embedding to tokens directly
        embed = self.rope(embed)

        # embed = self.norm(embed)
        # -> (batch, stream, 2 (k/v), n_heads, d_embed)
        kv_merged = self.kv_linear(embed) \
            .unflatten(-1, (2, self.n_attention_heads, self.d_embed))
        q = self.init_q
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # -> (batch, stream, n_heads, d_embed)
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)
        attn_concat = attn_out.flatten(-2, -1)
        # ff_in: (batch, stream, n_heads * d_embed)
        ff_mid_1, ff_mid_2 = self.ff_in(attn_concat).split(self.d_ff_inner, dim=-1)
        ff_mid = self.activation(ff_mid_1) * ff_mid_2
        ff_out = self.ff_out(ff_mid)

        return ff_out

class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rope = RotaryEncoding(config.d_embed, 128)

    def forward(self, x: torch.Tensor):
        pass

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

class BatchGLU(nn.Module):
    def __init__(self, n_streams: int, d_in: int, d_mid: int, d_out: int, activation, has_bias = True):
        super().__init__()
        self.d_mid = d_mid
        self.w_in = BatchLinear(n_streams, d_in, d_mid * 2, has_bias=has_bias)
        self.activation = activation
        self.w_out = BatchLinear(n_streams, d_mid, d_out, has_bias=has_bias)

    def forward(self, x: torch.Tensor):
        mid_1, mid_2 = self.w_in(x).split(self.d_mid, dim=-1)
        mid = self.activation(mid_1) * mid_2
        return self.w_out(mid)

def apply_inline_gate(x: torch.Tensor, gate_multiplier: float):
    # (..., d_embed + 1) -> (..., d_embed)
    resid_gate = F.sigmoid(x[..., -1]) * gate_multiplier
    return x[..., :-1] * resid_gate.unsqueeze(-1)

class CrossAttentionInput(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_attention_heads = config.n_attention_heads
        self.resid_gate_multiplier = config.resid_gate_multiplier

        self.input_embedding = nn.Embedding(config.vocab_size, config.d_embed, dtype=config.dtype)
        # marker for uncommitted
        self.uncommited_marker = nn.Parameter(torch.randn((config.d_embed,)))
        self.norm = nn.LayerNorm((config.d_embed))
        self.q_linear = BatchLinear(
            config.n_streams,
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            has_bias=config.qkv_bias,
        )
        self.rope = RotaryEncoding(config.d_embed, config.n_streams)
        self.kv_linear = nn.Linear(
            config.d_embed,
            2 * config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads + 1,)))

        ff_in_dim = config.d_embed * (config.n_attention_heads + 1)
        self.feedforward = BatchGLU(
            n_streams=config.n_streams,
            d_in=ff_in_dim,
            d_mid=config.d_ff_inner,
            d_out=config.d_embed + 1,
            activation=config.get_activation(),
        )

    def forward(self, current: torch.Tensor, inputs: torch.Tensor, uncommitted: torch.Tensor):
        # (batch, stream) -> (batch, stream, d_embed)
        inputs = self.input_embedding(inputs)
        # mark uncommitted positions
        inputs += torch.where(uncommitted.unsqueeze(-1), self.uncommited_marker, 0)
        # apply rotary embedding to byte embeddings directly
        inputs = self.rope(inputs)

        q = self.q_linear(self.norm(current)) \
            .unflatten(-1, (self.n_attention_heads, self.d_embed)) \
            .transpose(-2, -3)
        # kv_merged: (batch, stream, 2 (k/v), n_heads, d_embed)
        kv_merged = self.kv_linear(inputs) \
            .unflatten(-1, (2, self.n_attention_heads, self.d_embed))
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # -> (batch, stream, n_heads, d_embed)
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)
        # add skip
        attn_out = torch.cat((attn_out, current.unsqueeze(-2)), dim=-2)
        # scale heads
        attn_out = self.head_scales.unsqueeze(-1) * attn_out

        attn_concat = attn_out.flatten(-2, -1)
        # attn_concat: (batch, stream, n_heads * d_embed + 1)
        ff_out = self.feedforward(attn_concat)

        resid = apply_inline_gate(ff_out, self.resid_gate_multiplier)
        return resid

class Intermediate(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_attention_heads = config.n_attention_heads
        self.resid_gate_multiplier = config.resid_gate_multiplier

        self.norm = nn.LayerNorm((config.d_embed))
        self.q_linear = BatchLinear(
            config.n_streams,
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            has_bias=config.qkv_bias,
        )
        self.kv_linear = nn.Linear(
            config.d_embed,
            2 * config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads + 1,)))

        ff_in_dim = config.d_embed * (config.n_attention_heads + 1)
        self.feedforward = BatchGLU(
            n_streams=config.n_streams,
            d_in=ff_in_dim,
            d_mid=config.d_ff_inner,
            d_out=config.d_embed + 1,
            activation=config.get_activation(),
        )

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        q = self.q_linear(x_norm) \
            .unflatten(-1, (self.n_attention_heads, self.d_embed)) \
            .transpose(-2, -3)
        kv_merged = self.kv_linear(x_norm) \
            .unflatten(-1, (2, self.n_attention_heads, self.d_embed))
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # -> (batch, stream, n_heads, d_embed)
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)
        # add skip
        attn_out = torch.cat((attn_out, x_norm.unsqueeze(-2)), dim=-2)
        # scale heads
        attn_out = self.head_scales.unsqueeze(-1) * attn_out

        attn_concat = attn_out.flatten(-2, -1)
        # attn_concat: (batch, stream, n_heads * d_embed + 1)
        ff_out = self.feedforward(attn_concat)

        resid = apply_inline_gate(ff_out, self.resid_gate_multiplier)
        return resid

class PreOutput(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_attention_heads = config.n_attention_heads

        self.norm = nn.LayerNorm((config.d_embed))
        self.q_linear = BatchLinear(
            config.n_streams,
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            has_bias=config.qkv_bias,
        )
        self.kv_linear = nn.Linear(
            config.d_embed,
            2 * config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads,)))

        # no skip
        ff_in_dim = config.d_embed * config.n_attention_heads
        self.feedforward = BatchGLU(
            n_streams=config.n_streams,
            d_in=ff_in_dim,
            d_mid=config.d_ff_inner,
            d_out=config.d_embed,
            activation=config.get_activation(),
        )

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        q = self.q_linear(x_norm) \
            .unflatten(-1, (self.n_attention_heads, self.d_embed)) \
            .transpose(-2, -3)
        kv_merged = self.kv_linear(x_norm) \
            .unflatten(-1, (self.n_attention_heads, self.d_embed))
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)
        # scale heads
        attn_out = self.head_scales.squeeze(-1) * attn_out
        attn_concat = attn_out.flatten(-2, -1)
        ff_out = self.feedforward(attn_concat)
        return ff_out

class CharDecode(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.query_base = nn.Parameter(torch.randn((config.d_embed,)))
        self.rope = RotaryEncoding(config.d_embed, 128)
        self.kv_linear = nn.Linear(config.d_embed, config.d_embed * 2)

        self.feedforward = GLU(config.d_embed, config.d_embed, config.get_activation())
        self.w_out = nn.Linear(config.d_embed, config.vocab_size)

    def forward(self, x: torch.Tensor, output_count: int):
        kv_merged = self.kv_linear(x).unflatten(-1, (2, self.d_embed))
        # kv_merged: (batch, stream, 2, n_embed)
        # no transpose since we have no n_heads
        k = kv_merged[..., 0, :]
        v = kv_merged[..., 1, :]
        # build queries: repeat query_base for output_count then apply encoding
        q = self.rope(self.query_base.unsqueeze(0).repeat_interleave(output_count, 0))

        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v)
        tokens_out = self.feedforward(attn_out)
        logits_out = self.w_out(tokens_out)
        return tokens_out, logits_out

class RNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.recurrent_init = nn.Parameter(torch.randn(config.n_streams, config.d_embed))
        self.input = CrossAttentionInput(config)
        self.intermediate = nn.ModuleList(
            Intermediate(config) for _ in range(config.n_intermediate)
        )
        self.pre_output = PreOutput(config)
        self.char_decode = CharDecode(config)

    def forward(
        self,
        recurrent: torch.Tensor,
        inputs: torch.Tensor,
        inputs_uncommitted: torch.Tensor,
        output_count: int,
    ):
        recurrent += self.input(recurrent, inputs, inputs_uncommitted)
        for layer in self.intermediate:
            recurrent += layer(recurrent)

        output = self.pre_output(recurrent)
        tokens_out, logits_out = self.char_decode(output, output_count)

        return recurrent, tokens_out, logits_out

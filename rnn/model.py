import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import ModelConfig
#from .rotary_encoding import RotaryEncoding

# def oh_no(x: torch.Tensor):
#     "oh no"
#     if x.abs().max() > 1e12:
#         print(x)
#         print('offending value:', x.abs().max())
#         raise RuntimeError('detected very large value(s)')
#     if not torch.all(torch.isfinite(x)).item():
#         if torch.any(torch.isnan(x)).item():
#             print('detected nan values')
#         if torch.any(torch.isinf(x)).item():
#             print('detected +/-inf values')
#
#         print(x)
#         raise RuntimeError('non-finite values detected')

class GLU(nn.Module):
    def __init__(self, in_dim, out_dim, activation, bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.activation = activation
        self.w = nn.Linear(in_dim, out_dim * 2, bias=bias)

    def forward(self, x: torch.Tensor):
        w1, w2 = self.w(x).split(self.out_dim, dim=-1)
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
            x = x + self.bias

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
        # print('mid_1 max:', mid_1.abs().max())
        # print('mid_2 max:', mid_2.abs().max())
        mid = self.activation(mid_1) * mid_2
        return self.w_out(mid)

class ReversedLinear(nn.Module):
    # A @ x + b, as opposed to the "normal" linear
    # what is this even supposed to be?
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.weight = nn.Parameter(torch.zeros((d_out, d_in)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor):
        return self.weight @ x

def apply_inline_gate(x: torch.Tensor, gate_multiplier: float):
    # (..., d_embed + 1) -> (..., d_embed)
    resid_gate = F.sigmoid(x[..., -1]) * gate_multiplier
    return x[..., :-1] * resid_gate.unsqueeze(-1)

class InputEncode(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_attention_heads = config.n_attention_heads

        self.input_embedding = nn.Embedding(config.vocab_size, config.d_embed)
        # marker for uncommitted
        self.uncommitted_marker = nn.Parameter(torch.randn((config.d_embed,)))
        self.positional_encodings = nn.Parameter(
            torch.randn((config.short_ctx_len + config.out_ctx_len, config.d_embed))
        )
        self.in_query = nn.Parameter(torch.randn((config.n_ext_streams, config.d_embed,)))
        self.kv_linear = nn.Linear(
            config.d_embed,
            2 * config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads,)))

        ff_in_dim = config.d_embed * config.n_attention_heads
        self.feedforward = BatchGLU(
            n_streams=config.n_ext_streams,
            d_in=ff_in_dim,
            d_mid=config.d_ff_inner,
            d_out=config.d_embed,
            activation=config.get_activation(),
        )

    def forward(self, inputs: torch.Tensor, committed: torch.Tensor):
        # (batch, stream) -> (batch, stream, d_embed)
        inputs_embed = self.input_embedding(inputs)
        # mark uncommitted positions
        inputs_embed = inputs_embed + torch.where(committed.unsqueeze(-1), 0, self.uncommitted_marker)
        # apply positional encoding to byte embeddings directly
        inputs_embed = inputs_embed + self.positional_encodings

        q = self.in_query
        # kv_merged: (batch, stream, 2 (k/v), n_heads, d_embed)
        kv_merged = self.kv_linear(inputs_embed) \
            .unflatten(-1, (2, self.n_attention_heads, self.d_embed))
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # -> (batch, stream, n_heads, d_embed)
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)
        # scale heads
        attn_out = self.head_scales.unsqueeze(-1) * attn_out

        attn_concat = attn_out.flatten(-2, -1)
        # attn_concat: (batch, stream, n_heads * d_embed)
        ff_out = self.feedforward(attn_concat)

        return ff_out

class PartialCrossAttention(nn.Module):
    def __init__(self, config: ModelConfig, a_streams: int, b_streams: int):
        super().__init__()
        self.a_streams = a_streams
        self.b_streams = b_streams
        self.total_streams = a_streams + b_streams
        self.n_attention_heads = config.n_attention_heads
        self.d_embed = config.d_embed

        self.q_linear = BatchLinear(
            self.a_streams,
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            has_bias=config.qkv_bias,
        )
        self.k_a_linear = nn.Linear(
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.k_b_linear = nn.Linear(
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.v1_a_linear = nn.Linear(config.d_embed, config.d_embed, bias=config.qkv_bias)
        self.v1_b_linear = nn.Linear(config.d_embed, config.d_embed, bias=config.qkv_bias)
        self.v2_linear = ReversedLinear(
            self.total_streams,
            config.n_attention_heads * self.total_streams,
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # a: self, b: other
        # query from self
        q = self.q_linear(a) \
            .unflatten(-1, (self.n_attention_heads, self.d_embed)) \
            .transpose(-2, -3)
        # keys from self
        k_a = self.k_a_linear(a).unflatten(-1, (self.n_attention_heads, self.d_embed))
        # keys from other
        k_b = self.k_b_linear(b).unflatten(-1, (self.n_attention_heads, self.d_embed))
        k = torch.cat((k_a, k_b), dim=-3).transpose(-2, -3)
        # transform for values
        v1_a = self.v1_a_linear(a)
        v1_b = self.v1_b_linear(b)
        # concatenate and apply "sequence mixing"
        v = self.v2_linear(torch.cat((v1_a, v1_b), dim=-2)) \
            .unflatten(-2, (self.n_attention_heads, self.total_streams))
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)
        return attn_out

class Intermediate(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_attention_heads = config.n_attention_heads
        self.resid_gate_multiplier = config.resid_gate_multiplier
        self.n_int_streams = config.n_int_streams
        self.n_ext_streams = config.n_ext_streams
        self.n_total_streams = config.n_int_streams + config.n_ext_streams

        self.int_norm = nn.LayerNorm([config.d_embed])
        self.ext_norm = nn.LayerNorm([config.d_embed])
        self.attn_int = PartialCrossAttention(config, self.n_int_streams, self.n_ext_streams)
        self.attn_ext = PartialCrossAttention(config, self.n_ext_streams, self.n_int_streams)
        self.head_scales_int = nn.Parameter(torch.ones((config.n_attention_heads + 1,)))
        self.head_scales_ext = nn.Parameter(torch.ones((config.n_attention_heads + 1,)))

        ff_in_dim = config.d_embed * (config.n_attention_heads + 1)
        self.feedforward = BatchGLU(
            n_streams=config.n_int_streams + config.n_ext_streams,
            d_in=ff_in_dim,
            d_mid=config.d_ff_inner,
            d_out=config.d_embed + 1,
            activation=config.get_activation(),
        )

    def forward(self, x_int: torch.Tensor, x_ext: torch.Tensor):
        x_int_norm = self.int_norm(x_int)
        x_ext_norm = self.ext_norm(x_ext)

        x_attn_int = self.attn_int(x_int_norm, x_ext_norm)
        x_attn_ext = self.attn_ext(x_ext_norm, x_int_norm)
        # -> (batch, stream, n_heads, d_embed)
        # add skip
        x_attn_int = torch.cat((x_attn_int, x_int_norm.unsqueeze(-2)), dim=-2)
        x_attn_ext = torch.cat((x_attn_ext, x_ext_norm.unsqueeze(-2)), dim=-2)
        # scale heads
        x_attn_int = self.head_scales_int.unsqueeze(-1) * x_attn_int
        x_attn_ext = self.head_scales_ext.unsqueeze(-1) * x_attn_ext

        # concat internal and external
        x_attn = torch.cat((x_attn_int, x_attn_ext), dim=-3)
        # concat heads
        x_attn_concat = x_attn.flatten(-2, -1)
        # x_attn_concat: (batch, stream, n_heads * d_embed + 1)
        ff_out = self.feedforward(x_attn_concat)

        resid = apply_inline_gate(ff_out, self.resid_gate_multiplier)
        # split internal/external
        resid_int, resid_ext = resid.split([self.n_int_streams, self.n_ext_streams], dim=-2)
        return resid_int, resid_ext

class OutputAdapter(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_attention_heads = config.n_attention_heads
        self.n_ext_streams = config.n_ext_streams

        self.norm = nn.LayerNorm([config.d_embed])
        self.q_linear = BatchLinear(
            config.n_ext_streams,
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            has_bias=config.qkv_bias,
        )
        self.k_linear = nn.Linear(
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.v_linear = ReversedLinear(
            config.n_ext_streams,
            config.n_attention_heads * config.n_ext_streams,
        )
        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads,)))

        # no skip
        ff_in_dim = config.d_embed * config.n_attention_heads
        self.feedforward = BatchGLU(
            n_streams=config.n_ext_streams,
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
        k = self.k_linear(x_norm) \
            .unflatten(-1, (self.n_attention_heads, self.d_embed)) \
            .transpose(-2, -3)
        v = self.v_linear(x_norm) \
            .unflatten(-2, (self.n_attention_heads, self.n_ext_streams))

        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)
        # scale heads
        attn_out = self.head_scales.unsqueeze(-1) * attn_out
        attn_concat = attn_out.flatten(-2, -1)
        ff_out = self.feedforward(attn_concat)
        return ff_out

class CharDecode(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.norm = nn.LayerNorm([config.d_embed])
        self.out_query = nn.Parameter(torch.randn((config.out_ctx_len, config.d_embed,)))
        self.k_linear = nn.Linear(config.d_embed, config.d_embed, bias=config.qkv_bias)
        self.v_linear = ReversedLinear(config.n_ext_streams, config.n_ext_streams)

        self.feedforward = GLU(config.d_embed, config.d_embed, config.get_activation())
        self.w_out = nn.Linear(config.d_embed, config.vocab_size)

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        # no transpose since we have no n_heads
        q = self.out_query
        k = self.k_linear(x_norm)
        v = self.v_linear(x_norm)

        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v)
        embeddings_out = self.feedforward(attn_out)
        logits_out = self.w_out(embeddings_out)
        return embeddings_out, logits_out

class RNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.recurrent_init = nn.Parameter(torch.zeros(config.n_int_streams, config.d_embed))
        self.input = InputEncode(config)
        self.intermediate = nn.ModuleList(
            Intermediate(config) for _ in range(config.n_intermediate)
        )
        self.output_adapter = OutputAdapter(config)
        self.char_decode = CharDecode(config)

        init_bound = 1 / math.sqrt(config.d_embed)
        nn.init.uniform_(self.recurrent_init, -init_bound, init_bound)

    def forward(
        self,
        recurrent: torch.Tensor,
        inputs: torch.Tensor,
        inputs_committed: torch.Tensor,
    ):
        external = self.input(inputs, inputs_committed)
        for layer in self.intermediate:
            recurrent_resid, external_resid = layer(recurrent, external)
            recurrent = recurrent + recurrent_resid
            external = external + external_resid

        external = self.output_adapter(external)
        embeddings_out, logits_out = self.char_decode(external)

        return recurrent, embeddings_out, logits_out

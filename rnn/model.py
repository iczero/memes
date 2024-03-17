import torch
from torch import nn
from torch.nn import functional as F

from .common import ModelConfig
from .rotary_encoding import RotaryEncoding


class PartialCrossAttention(nn.Module):
    "Partial cross attention layer"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_attention_heads = config.n_attention_heads
        self.n_embed = config.n_embed
        self.attn_dropout_p = config.attn_dropout_p
        # in: external: (batch, seq, n_embed), internal: (batch, seq, n_embed)
        # split these two since q and k/v have different inputs
        self.q_linear = nn.Linear(
            config.n_embed,
            config.n_attention_heads * config.n_embed,
            bias=config.qkv_bias,
        )
        self.kv_linear = nn.Linear(
            config.n_embed,
            2 * config.n_attention_heads * config.n_embed,
            bias=config.qkv_bias,
        )

    def forward(self, external: torch.Tensor, internal: torch.Tensor):
        # external is the other sequence concatenated to k/v, internal is our own sequence
        # q shape: (batch, seq, n_heads, n_embed)
        # transpose (batch, seq, n_heads, n_embed) -> (batch, n_heads, seq, n_embed) for sdp
        q = self.q_linear(internal) \
            .unflatten(-1, (self.n_attention_heads, self.n_embed)) \
            .transpose(-2, -3)
        kv_seq = torch.concat((external, internal), dim=-2)
        # kv_merged shape: (batch, seq, 2, n_heads, n_embed)
        kv_merged = self.kv_linear(kv_seq).unflatten(-1, (2, self.n_attention_heads, self.n_embed))
        # extract from merged
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # why does pylint think this isn't callable?
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout_p)

        # transpose back
        return attn_out.transpose(-2, -3)

class GatedFeedForward(nn.Module):
    "Feedforward after attention with residual gating"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.register_buffer('resid_gate_multiplier', torch.tensor(config.resid_gate_multiplier))
        in_len = config.n_embed * config.n_attention_heads
        out_len = config.n_embed + 1
        # in: (batch, seq, n_heads, n_embed) -> (batch, seq, n_heads * n_embed)
        # out: residuals (batch, seq, n_embed) after gating
        # modified feedforward: dense after heads, then rescale back to n_embed
        self.stack = nn.Sequential(
            # concat all heads
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(in_len, in_len),
            config.activation(),
            nn.Linear(in_len, out_len),
        )
        self.dropout = nn.Dropout(config.ff_dropout_p)

    def forward(self, x):
        out = self.stack(x)
        # extract resid gate
        resid_gate = F.sigmoid(out[..., -1]) * self.resid_gate_multiplier
        # calculate residual, apply dropout
        resid = self.dropout(out[..., :-1])
        # gate residual and return
        return resid * resid_gate.unsqueeze(-1)

class InputLayer(nn.Module):
    "The input layer of the model"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_dropout_p = config.attn_dropout_p
        self.short_ctx_len = config.short_ctx_len
        # in: (batch, seq)
        self.input_embedding = nn.Embedding(config.vocab_size, config.n_embed)

        self.make_qkv = nn.Sequential(
            self.input_embedding,
            # qkv linear, (batch, seq, n_embed) -> (batch, seq, 3 * n_heads * n_embed)
            nn.Linear(
                config.n_embed,
                3 * config.n_attention_heads * config.n_embed,
                bias=config.qkv_bias,
            ),
            # (batch, seq, n_embed) -> (batch, seq, 3, n_heads, n_embed)
            nn.Unflatten(-1, (3, config.n_attention_heads, config.n_embed))
        )

        # positional encoding
        self.rope = RotaryEncoding(config.n_embed, config.short_ctx_len)

        # no residuals in input layer
        # ff in: (batch, seq, n_heads, n_embed) -> (batch, seq, n_heads * n_embed)
        ff_in_dim = config.n_embed * config.n_attention_heads
        self.feedforward = nn.Sequential(
            # concat heads
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(ff_in_dim, ff_in_dim),
            config.activation(),
            nn.Linear(ff_in_dim, config.n_embed),
            nn.Dropout(config.ff_dropout_p),
        )

    def forward(self, x: torch.Tensor):
        # ensure sequence is of correct length
        assert x.shape[-1] == self.short_ctx_len

        qkv_merged: torch.Tensor = self.make_qkv(x)
        # extract q/k/v from merged
        # (batch, seq, n_heads, n_embed)
        # transpose to (batch, n_heads, seq, n_embed) for sdp
        q = qkv_merged[..., 0, :, :].transpose(-2, -3)
        k = qkv_merged[..., 1, :, :].transpose(-2, -3)
        v = qkv_merged[..., 2, :, :].transpose(-2, -3)

        # apply rotary positional embedding on q/k
        q = self.rope(q)
        k = self.rope(k)

        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout_p)

        # transpose back (batch, seq, n_heads, n_embed), then ff
        # out shape is (batch, seq, n_embed)
        return self.feedforward(attn_out.transpose(-2, -3))

class IntermediateLayer(nn.Module):
    "The intermediate layers(s) of the model"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm((config.n_embed,))
        self.attention = PartialCrossAttention(config)
        self.feedforward = GatedFeedForward(config)

    def forward(self, recurrent: torch.Tensor, internal: torch.Tensor):
        bypass = internal
        internal = self.norm(internal)
        internal = self.attention(recurrent, internal)
        internal = self.feedforward(internal)
        return bypass + internal

class OutputLayer(nn.Module):
    "The output layer of the model"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm((config.n_embed,))
        self.attention = PartialCrossAttention(config)
        self.feedforward = GatedFeedForward(config)

    def forward(self, recurrent: torch.Tensor, internal: torch.Tensor):
        # normalize internal from previous intermediate layer
        internal = self.norm(internal)
        # we are computing residual to recurrent state, so "recurrent" is "internal"
        out = self.attention(internal, recurrent)
        out = self.feedforward(out)
        return out

class RNNPonder(nn.Module):
    """
    The part of the model rerun for ponder

    Contains intermediate layers followed by an output layer. Does not contain
    the input layer, which should be run before.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm((config.n_embed,))
        self.intermediate = nn.ModuleList(
            IntermediateLayer(config) for _ in range(config.n_intermediate))
        self.output = OutputLayer(config)

    # recurrent is recurrent state, internal is output of input layer, or if
    # pondering, the internal output of the previous RNNPonder step
    def forward(self, recurrent: torch.Tensor, internal: torch.Tensor):
        bypass = recurrent
        recurrent = self.norm(recurrent)
        for layer in self.intermediate:
            internal = layer(recurrent, internal)

        resid = self.output(recurrent, internal)
        out = bypass + resid
        return out, internal

class OutputDecode(nn.Module):
    "Derive output token and ponder p_halt from recurrent state"
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_attention_heads = config.n_attention_heads
        self.n_embed = config.n_embed

        self.norm = nn.LayerNorm((config.n_embed,))
        # standard cross-attention scheme except queries are directly parameters
        self.q_out = nn.Parameter(torch.randn(config.n_embed))
        self.q_p_halt = nn.Parameter(torch.randn(config.n_embed))
        self.kv_linear = nn.Linear(
            config.n_embed,
            2 * config.n_attention_heads * config.n_embed,
            bias=config.qkv_bias,
        )

        ff_in_dim = config.n_embed * config.n_attention_heads
        self.out_feedforward = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(ff_in_dim, ff_in_dim),
            config.activation(),
            nn.Linear(ff_in_dim, config.n_embed),
            nn.Linear(config.n_embed, config.vocab_size),
        )

        self.p_halt_feedforward = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(ff_in_dim, ff_in_dim),
            config.activation(),
            nn.Linear(ff_in_dim, 1), # p_halt
        )

    def forward(self, recurrent: torch.Tensor):
        # (batch, "seq", n_embed)
        q = torch.stack((self.q_out, self.q_p_halt)).unsqueeze(0)
        recurrent = self.norm(recurrent)
        kv_merged = self.kv_linear(recurrent) \
            .unflatten(-1, (2, self.n_attention_heads, self.n_embed))
        # extract and transpose for sdp
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # no dropout here
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(-2, -3)
        # (batch, "seq", n_heads, n_embed) -> (batch, n_embed)
        token_out = self.out_feedforward(attn_out[..., 0, :, :])
        # -> (batch, 1)
        p_halt_out = self.p_halt_feedforward(attn_out[..., 1, :, :])

        # out: (batch, vocab_size), (batch)
        return token_out, F.sigmoid(p_halt_out.squeeze(-1))

class RNNSequence(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.recurrent_init = nn.Parameter(torch.randn((config.recurrent_seq_len, config.n_embed)))
        self.input = InputLayer(config)
        self.ponder = RNNPonder(config)
        self.decode = OutputDecode(config)

    def forward(self):
        # this module probably needs more than just forward()
        raise NotImplementedError()

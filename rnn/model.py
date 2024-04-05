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
        # https://arxiv.org/pdf/2110.09456.pdf
        # TODO: go make references section
        self.head_scales = nn.Parameter(torch.zeros((config.n_attention_heads,)))

    def forward(self, internal: torch.Tensor, external: torch.Tensor):
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
        attn_out = attn_out.transpose(-2, -3)

        # scale heads
        return (1 + self.head_scales).unsqueeze(-1) * attn_out

class SelfAttention(nn.Module):
    "Self attention layer with positional encoding"

    def __init__(self, config: ModelConfig, is_input_layer=False):
        super().__init__()
        self.n_attention_heads = config.n_attention_heads
        self.n_embed = config.n_embed
        # in: (batch, seq, n_embed)
        self.qkv_linear = nn.Linear(
            config.n_embed,
            3 * config.n_attention_heads * config.n_embed,
            bias=config.qkv_bias,
        )

        if is_input_layer:
            # use rotary positional encoding
            self.rope = RotaryEncoding(config.n_embed, config.short_ctx_len)
            # do not use dropout in input layer (does this make a difference?)
            self.attn_dropout = 0
        else:
            self.rope = None
            self.attn_dropout = config.attn_dropout_p

    def forward(self, sequence: torch.Tensor):
        # kv_merged shape: (batch, seq, 3 (q/k/v), n_heads, n_embed)
        kv_merged = self.qkv_linear(sequence) \
            .unflatten(-1, (3, self.n_attention_heads, self.n_embed))
        # extract from merged
        q = kv_merged[..., 0, :, :].transpose(-2, -3)
        k = kv_merged[..., 1, :, :].transpose(-2, -3)
        v = kv_merged[..., 2, :, :].transpose(-2, -3)

        if self.rope is not None:
            # apply positional encodings
            q = self.rope(q)
            k = self.rope(k)

        # why does pylint think this isn't callable?
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout)

        # transpose back
        return attn_out.transpose(-2, -3)

class GatedFeedForward(nn.Module):
    "Feedforward after attention with residual gating"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.register_buffer('resid_gate_multiplier', torch.tensor(config.resid_gate_multiplier))
        in_dim = config.n_embed * config.n_attention_heads
        mid_dim = config.ff_dim_multiplier * config.n_embed
        out_dim = config.n_embed + 1
        # in: (batch, seq, n_heads, n_embed) -> (batch, seq, n_heads * n_embed)
        # out: residuals (batch, seq, n_embed) after gating
        # modified feedforward: dense after heads, then rescale back to n_embed
        self.stack = nn.Sequential(
            # concat all heads
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(in_dim, mid_dim),
            config.get_activation(),
            nn.Linear(mid_dim, mid_dim),
            config.get_activation(),
            nn.Dropout(config.ff_dropout_p),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x):
        out = self.stack(x)
        # extract resid gate
        resid_gate = F.sigmoid(out[..., -1]) * self.resid_gate_multiplier
        # calculate residual
        resid = out[..., :-1]
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
        self.attention = SelfAttention(config, is_input_layer=True)

        # no residuals in input layer
        # ff in: (batch, seq, n_heads, n_embed) -> (batch, seq, n_heads * n_embed)
        ff_in_dim = config.n_embed * config.n_attention_heads
        ff_mid_dim = config.ff_dim_multiplier * config.n_embed
        self.feedforward = nn.Sequential(
            # concat heads
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(ff_in_dim, ff_mid_dim),
            config.get_activation(),
            nn.Linear(ff_mid_dim, ff_mid_dim),
            config.get_activation(),
            nn.Dropout(config.ff_dropout_p),
            nn.Linear(ff_mid_dim, config.n_embed),
        )

    def forward(self, x: torch.Tensor):
        # ensure sequence is of correct length
        #assert x.shape[-1] == self.short_ctx_len

        embeddings = self.input_embedding(x)
        attn_out = self.attention(embeddings)

        # out shape is (batch, seq, n_embed)
        return self.feedforward(attn_out)

class IntermediateLayer(nn.Module):
    "The intermediate layers(s) of the model"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm((config.n_embed,))
        self.recurrent_attention = PartialCrossAttention(config)
        self.recurrent_feedforward = GatedFeedForward(config)
        self.internal_attention = PartialCrossAttention(config)
        self.internal_feedforward = GatedFeedForward(config)

    def forward(self, recurrent: torch.Tensor, internal: torch.Tensor):
        recurrent_norm = self.norm(recurrent)
        internal_norm = self.norm(internal)

        recurrent_resid = self.recurrent_attention(recurrent_norm, internal_norm)
        recurrent_resid = self.recurrent_feedforward(recurrent_resid)

        internal_resid = self.internal_attention(internal_norm, recurrent_norm)
        internal_resid = self.internal_feedforward(internal_resid)

        return recurrent + recurrent_resid, internal + internal_resid

class OutputDecode(nn.Module):
    "Derive output token and ponder confidence from internal state"
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_embed = config.n_embed

        self.norm = nn.LayerNorm((config.n_embed,))

        # standard cross-attention scheme except queries are directly parameters
        self.q_out = nn.Parameter(torch.randn(config.n_embed))
        self.kv_linear_out = nn.Linear(
            config.n_embed,
            2 * config.n_embed,
            bias=config.qkv_bias,
        )

        self.q_confidence = nn.Parameter(torch.randn(config.n_embed))
        self.kv_linear_confidence = nn.Linear(
            config.n_embed,
            2 * config.n_embed,
            bias=config.qkv_bias,
        )

        self.ff_out = nn.Sequential(
            nn.Linear(config.n_embed, config.ff_dim_multiplier * config.n_embed),
            config.get_activation(),
            nn.Linear(config.ff_dim_multiplier * config.n_embed, config.n_embed),
            nn.Linear(config.n_embed, config.vocab_size),
        )

        self.ff_confidence = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed),
            config.get_activation(),
            nn.Linear(config.n_embed, 1), # confidence
        )

    def forward(self, internal: torch.Tensor):
        # in: (batch, seq, n_embed)
        internal = self.norm(internal)

        q = torch.stack((
            self.q_out.unsqueeze(0),
            self.q_confidence.unsqueeze(0),
        ), dim=0)
        # -> (out/halt, "seq", n_embed)

        kv_merged = torch.stack((
            self.kv_linear_out(internal).unflatten(-1, (2, self.n_embed)),
            self.kv_linear_confidence(internal).unflatten(-1, (2, self.n_embed)),
        ), dim=-4)
        # kv_merged: (batch, out/halt, seq, k/v, n_embed)
        # extract k/v for sdp
        k = kv_merged[..., 0, :]
        v = kv_merged[..., 1, :]
        # -> (batch, out/halt, seq, n_embed)

        # no dropout here
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v)

        # (batch, out/halt, "seq", n_embed) -> (batch, n_embed)
        token_out = self.ff_out(attn_out[..., 0, 0, :])
        # -> (batch, 1)
        confidence_out = self.ff_confidence(attn_out[..., 1, 0, :])

        # out: (batch, vocab_size), (batch)
        return token_out, confidence_out.squeeze(-1)

class RNNPonder(nn.Module):
    """
    The part of the model rerun for ponder

    Contains intermediate layers followed by the decode layer. Does not contain
    the input layer, which should be run before.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.short_ctx_len = config.short_ctx_len
        self.recurrent_seq_len = config.recurrent_seq_len

        self.intermediate = nn.ModuleList(
            IntermediateLayer(config) for _ in range(config.n_intermediate))
        self.output = OutputDecode(config)

    # recurrent is recurrent state, internal is output of input layer, or if
    # pondering, the internal output of the previous RNNPonder step
    def forward(self, recurrent: torch.Tensor, internal: torch.Tensor):
        # in: (batch, seq, n_embed)
        for layer in self.intermediate:
            recurrent, internal = layer(recurrent, internal)

        token_out, confidence_out = self.output(internal)
        return recurrent, internal, token_out, confidence_out

class RNNSequence(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.recurrent_seq_len = config.recurrent_seq_len
        self.recurrent_init = nn.Parameter(torch.randn(config.n_embed))
        self.recurrent_init_rope = RotaryEncoding(config.n_embed, config.recurrent_seq_len)
        self.input = InputLayer(config)
        self.ponder = RNNPonder(config)

    def make_recurrent_init(self):
        recurrent_init = self.recurrent_init.unsqueeze(0) \
            .repeat_interleave(self.recurrent_seq_len, 0)
        return self.recurrent_init_rope(recurrent_init)

    def forward(self):
        # this module probably needs more than just forward()
        raise NotImplementedError()

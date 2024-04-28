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
        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads + 1,)))

    def forward(self, internal_norm: torch.Tensor, internal_resid: torch.Tensor, external_norm: torch.Tensor):
        # external is the other sequence concatenated to k/v, internal is our own sequence
        # q shape: (batch, seq, n_heads, n_embed)
        # transpose (batch, seq, n_heads, n_embed) -> (batch, n_heads, seq, n_embed) for sdp
        q = self.q_linear(internal_norm) \
            .unflatten(-1, (self.n_attention_heads, self.n_embed)) \
            .transpose(-2, -3)
        kv_seq = torch.concat((external_norm, internal_norm), dim=-2)
        # kv_merged shape: (batch, seq, 2, n_heads, n_embed)
        kv_merged = self.kv_linear(kv_seq).unflatten(-1, (2, self.n_attention_heads, self.n_embed))
        # extract from merged
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # why does pylint think this isn't callable?
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout_p)

        # transpose back
        # -> (batch, seq, n_heads, n_embed)
        attn_out = attn_out.transpose(-2, -3)

        # concat skip connection
        # -> (batch, seq, n_heads + 1, n_embed)
        attn_out = torch.cat((attn_out, internal_resid.unsqueeze(-2)), dim=-2)

        # scale heads
        return self.head_scales.unsqueeze(-1) * attn_out

class GatedFeedForward(nn.Module):
    "Feedforward after attention with residual gating"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.register_buffer('resid_gate_multiplier', torch.tensor(config.resid_gate_multiplier))
        in_dim = config.n_embed * (config.n_attention_heads + 1)
        out_dim = config.n_embed + 1
        # in: (batch, seq, n_heads + 1, n_embed) -> (batch, seq, (n_heads + 1) * n_embed)
        # out: residuals (batch, seq, n_embed) after gating
        # modified feedforward: dense after heads, then rescale back to n_embed
        self.stack = nn.Sequential(
            # concat all heads
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(in_dim, config.ff_inner_dim),
            config.get_activation(),
            nn.Linear(config.ff_inner_dim, config.ff_inner_dim),
            config.get_activation(),
            nn.Dropout(config.ff_dropout_p),
            nn.Linear(config.ff_inner_dim, out_dim),
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
        self.n_attention_heads = config.n_attention_heads
        self.n_embed = config.n_embed

        # in: (batch, seq)
        self.input_embedding = nn.Embedding(config.vocab_size, config.n_embed)

        self.norm = nn.LayerNorm((config.n_embed,))

        # in: (batch, seq, n_embed)
        self.qkv_linear = nn.Linear(
            config.n_embed,
            3 * config.n_attention_heads * config.n_embed,
            bias=config.qkv_bias,
        )

        # use rotary positional encoding
        self.rope = RotaryEncoding(config.n_embed, config.short_ctx_len)

        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads + 1,)))

        # no residuals in input layer
        # ff in: (batch, seq, n_heads + 1, n_embed) -> (batch, seq, (n_heads + 1) * n_embed + 1)
        # one more input for new_mask
        ff_in_dim = config.n_embed * (config.n_attention_heads + 1) + 1
        self.feedforward = nn.Sequential(
            nn.Linear(ff_in_dim, config.ff_inner_dim),
            config.get_activation(),
            nn.Linear(config.ff_inner_dim, config.ff_inner_dim),
            config.get_activation(),
            nn.Dropout(config.ff_dropout_p),
            nn.Linear(config.ff_inner_dim, config.n_embed),
        )

    def forward(self, tokens: torch.Tensor, new_mask: torch.Tensor):
        # ensure sequence is of correct length
        #assert x.shape[-1] == self.short_ctx_len

        embeddings = self.input_embedding(tokens)

        # kv_merged shape: (batch, seq, 3 (q/k/v), n_heads, n_embed)
        kv_merged = self.qkv_linear(self.norm(embeddings)) \
            .unflatten(-1, (3, self.n_attention_heads, self.n_embed))
        # extract from merged
        q = kv_merged[..., 0, :, :].transpose(-2, -3)
        k = kv_merged[..., 1, :, :].transpose(-2, -3)
        v = kv_merged[..., 2, :, :].transpose(-2, -3)

        # apply positional encoding
        q = self.rope(q)
        k = self.rope(k)

        # why does pylint think this isn't callable?
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout_p)

        # transpose back
        attn_out = attn_out.transpose(-2, -3)
        # concat skip
        attn_out = torch.cat((attn_out, embeddings.unsqueeze(-2)), dim=-2)
        # scale heads
        attn_out = self.head_scales.unsqueeze(-1) * attn_out

        # concat heads and add new token mark
        ff_in = torch.cat((
            attn_out.flatten(-2, -1),
            torch.where(new_mask, 1., 0.).unsqueeze(-1)
        ), dim=-1)

        # out shape is (batch, seq, n_embed)
        return self.feedforward(ff_in)

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

        recurrent_resid = self.recurrent_attention(recurrent_norm, recurrent, internal_norm)
        recurrent_resid = self.recurrent_feedforward(recurrent_resid)

        internal_resid = self.internal_attention(internal_norm, internal, recurrent_norm)
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
            nn.Linear(config.n_embed, config.ff_inner_dim),
            config.get_activation(),
            nn.Linear(config.ff_inner_dim, config.n_embed),
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

class RNNSequence(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.recurrent_seq_len = config.recurrent_seq_len
        self.recurrent_init = nn.Parameter(torch.randn(config.n_embed))
        self.recurrent_init_rope = RotaryEncoding(config.n_embed, config.recurrent_seq_len)

        self.input = InputLayer(config)
        self.intermediate = nn.ModuleList(
            IntermediateLayer(config) for _ in range(config.n_intermediate))
        self.output = OutputDecode(config)

    def make_recurrent_init(self):
        recurrent_init = self.recurrent_init.unsqueeze(0) \
            .repeat_interleave(self.recurrent_seq_len, 0)
        return self.recurrent_init_rope(recurrent_init)

    def forward(self, recurrent: torch.Tensor, short_ctx: torch.Tensor, new_mask: torch.Tensor):
        internal = self.input(short_ctx, new_mask)
        for layer in self.intermediate:
            recurrent, internal = layer(recurrent, internal)
        token_out, confidence_out = self.output(internal)
        return recurrent, token_out, confidence_out

import dataclasses
import functools
from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .rotary_encoding import RotaryEncoding


@dataclass
class ModelConfig:
    "Embedding dimensions"
    n_embed: int
    "Number of attention heads"
    n_attention_heads: int
    "Vocabulary size"
    vocab_size: int
    "Length of input short context sequence"
    short_ctx_len: int
    "Length of internal sequence"
    internal_seq_len: int
    "Length of recurrent sequence"
    recurrent_seq_len: int = dataclasses.field(init=False)
    "Probability of dropout after feedforward"
    ff_dropout_p: float
    "Probability of dropout after attention"
    attn_dropout_p: float
    "Number of intermediate layers"
    n_intermediate: int
    "Ponder: static penalty for pondering"
    ponder_continue_penalty: float
    "Ponder: penalty for halting with loss"
    ponder_loss_penalty: float
    "Multiplier for residual gating"
    resid_gate_multiplier: float
    "Activation function"
    activation: Callable[[], nn.Module]
    "Whether Q/K/V linear layers in attention should have bias"
    qkv_bias: bool

    def __post_init__(self):
        assert self.n_embed > 0
        assert self.n_attention_heads > 0
        assert self.vocab_size > 0
        assert self.short_ctx_len > 0
        assert self.internal_seq_len > 0
        assert self.internal_seq_len > self.short_ctx_len
        self.recurrent_seq_len = self.internal_seq_len - self.short_ctx_len
        assert self.ff_dropout_p >= 0
        assert self.ponder_loss_penalty >= 1
        assert self.ponder_continue_penalty >= 0
        assert self.resid_gate_multiplier > 0

    @classmethod
    def default(cls):
        return cls(
            n_embed=128,
            n_attention_heads=8,
            vocab_size=1, # TODO: train tokenizer
            short_ctx_len=64,
            internal_seq_len=256,
            ff_dropout_p=0.0,
            attn_dropout_p=0.0,
            n_intermediate=3,
            ponder_continue_penalty=10.0,
            ponder_loss_penalty=1.1,
            resid_gate_multiplier=2.0,
            activation=torch.nn.LeakyReLU,
            qkv_bias=True,
        )

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
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_dropout_p = config.attn_dropout_p
        # in: (batch, seq)
        self.input_embedding = nn.Embedding(config.vocab_size, config.n_embed)

        self.make_qkv = nn.Sequential(
            self.input_embedding,
            # (batch, seq, n_embed) -> (batch, seq, 3 * n_heads * n_embed)
            nn.Linear(config.n_embed, 3 * config.n_heads * config.n_embed, bias=config.qkv_bias),
            # (batch, seq, n_embed) -> (batch, seq, 3, n_heads, n_embed)
            nn.Unflatten(-1, (3, config.n_heads, config.n_embed))
        )

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

    def forward(self, x):
        qkv_merged: torch.Tensor = self.make_qkv(x)
        # extract q/k/v from merged
        # (batch, seq, n_heads, n_embed)
        # transpose to (batch, n_heads, seq, n_embed) for sdp
        q = qkv_merged[..., 0, :, :].transpose(1, 2)
        k = qkv_merged[..., 1, :, :].transpose(1, 2)
        v = qkv_merged[..., 2, :, :].transpose(1, 2)

        # ??????
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout_p)

        # transpose back (batch, seq, n_heads, n_embed), then ff
        return self.feedforward(attn_out.transpose(1, 2))

import dataclasses
import json
import warnings
from collections.abc import Callable

import numpy as np
import sentencepiece as spm
import zstandard
from torch import nn


@dataclasses.dataclass
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
            vocab_size=32000,
            short_ctx_len=64,
            internal_seq_len=256,
            ff_dropout_p=0.0,
            attn_dropout_p=0.0,
            n_intermediate=4,
            ponder_continue_penalty=5.0,
            ponder_loss_penalty=1.5,
            resid_gate_multiplier=2.0,
            activation=nn.GELU,
            qkv_bias=True,
        )

@dataclasses.dataclass
class TrainConfig:
    "Learning rate"
    lr: float
    "Weight decay"
    weight_decay: float
    "Probability to introduce bad token and backspace"
    backspace_p: float
    "Batch size"
    batch_size: int
    "Max steps to run training"
    # TODO: truncated BPTT or activation checkpointing or something
    max_steps: int

    @classmethod
    def default(cls):
        return cls(
            lr=0.001,
            weight_decay=0.001,
            backspace_p=0.01,
            batch_size=32,
            max_steps=128,
        )

def load_dataset(in_stream):
    dctx = zstandard.ZstdDecompressor()
    count = 0
    buffer = bytearray()
    for chunk in dctx.read_to_iter(in_stream):
        buffer += chunk
        lines = buffer.split(b'\n')
        buffer[:] = lines[-1]
        for line in lines[0:-1]:
            obj = json.loads(line)
            yield count, obj['text']
            count += 1

    if len(buffer) > 0:
        warnings.warn('dataset file did not end with newline')

def random_token_not(total: int, not_token: int):
    "Generate a random token id that is not the provided token"
    while True:
        token = np.random.randint(0, total)
        if token != not_token:
            return token

def tokenize_input(
    sp: spm.SentencePieceProcessor,
    ctx_len: int,
    sequence: str,
    train=False,
    backspace_p: float | None = None,
):
    pad_start = [sp['<pad>']] * (ctx_len - 1) + [sp['<s>']]
    last = [sp['</s>']]
    total_tokens = len(sp)
    encoded = sp.encode(sequence, out_type=int)

    if train:
        # randomly intersperse bad tokens and backspace
        where_backspace, = np.random.poisson(backspace_p, len(encoded)).nonzero()
        offset = 0
        for index in where_backspace:
            index += offset
            next_token = encoded[index]
            rand_token = random_token_not(total_tokens, next_token)
            encoded[index:index] = [rand_token, sp['<del>']]
            offset += 2

    return pad_start + encoded + last

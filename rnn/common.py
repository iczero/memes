import dataclasses
import json
import warnings
from collections.abc import Callable

import numpy as np
import zstandard
from sentencepiece import SentencePieceProcessor
from torch import nn


@dataclasses.dataclass
class ModelConfig:
    n_embed: int
    "Embedding dimensions"
    n_attention_heads: int
    "Number of attention heads"
    vocab_size: int
    "Vocabulary size"
    short_ctx_len: int
    "Length of input short context sequence"
    internal_seq_len: int
    "Length of internal sequence"
    recurrent_seq_len: int = dataclasses.field(init=False)
    "Length of recurrent sequence"
    ff_dropout_p: float
    "Probability of dropout after feedforward"
    attn_dropout_p: float
    "Probability of dropout after attention"
    n_intermediate: int
    "Number of intermediate layers"
    ponder_continue_penalty: float
    "Ponder: static penalty for pondering"
    ponder_loss_penalty: float
    "Ponder: penalty for halting with loss"
    resid_gate_multiplier: float
    "Multiplier for residual gating"
    activation: Callable[[], nn.Module]
    "Activation function"
    qkv_bias: bool
    "Whether Q/K/V linear layers in attention should have bias"

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
            n_attention_heads=4,
            vocab_size=32000,
            short_ctx_len=16,
            internal_seq_len=64,
            ff_dropout_p=0.0,
            attn_dropout_p=0.0,
            n_intermediate=3,
            #ponder_continue_penalty=6.0,
            #ponder_loss_penalty=1.7,
            # temporarily disable ponder
            ponder_continue_penalty=5.0,
            ponder_loss_penalty=1.0,
            resid_gate_multiplier=2.0,
            activation=nn.GELU,
            qkv_bias=True,
        )

@dataclasses.dataclass
class TrainConfig:
    lr: float
    "Learning rate"
    weight_decay: float
    "Weight decay"
    backspace_p: float
    "Probability to introduce bad token and backspace"
    batch_size: int
    "Batch size"
    max_ponder_steps: int
    "Max steps the model may ponder for"
    # TODO: truncated BPTT or activation checkpointing or something
    max_steps_temp: int
    "Max steps to run training (temporary)"

    @classmethod
    def default(cls):
        return cls(
            lr=0.001,
            weight_decay=0.001,
            backspace_p=0.01,
            batch_size=16,
            max_ponder_steps=8,
            max_steps_temp=16,
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
    sp: SentencePieceProcessor,
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

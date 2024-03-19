import dataclasses
import json
import struct
import warnings
from typing import Self

import torch
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
    "Ponder: penalty for halting with loss, overwritten during training"
    resid_gate_multiplier: float
    "Multiplier for residual gating"
    activation: str
    "Activation function"
    dtype: str
    "Data type of model"
    qkv_bias: bool
    "Whether Q/K/V linear layers in attention should have bias"
    tokenizer_model_path: str
    "Path to the tokenizer model"

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
    def from_dict(cls, obj: dict) -> Self:
        return cls(
            n_embed=int(obj['n_embed']),
            n_attention_heads=int(obj['n_attention_heads']),
            vocab_size=int(obj['vocab_size']),
            short_ctx_len=int(obj['short_ctx_len']),
            internal_seq_len=int(obj['internal_seq_len']),
            ff_dropout_p=float(obj['ff_dropout_p']),
            attn_dropout_p=float(obj['attn_dropout_p']),
            n_intermediate=int(obj['n_intermediate']),
            ponder_continue_penalty=float(obj['ponder_continue_penalty']),
            ponder_loss_penalty=float(obj['ponder_loss_penalty']),
            resid_gate_multiplier=float(obj['resid_gate_multiplier']),
            activation=str(obj['activation']),
            dtype=str(obj['dtype']),
            qkv_bias=bool(obj['qkv_bias']),
            tokenizer_model_path=str(obj['tokenizer_model_path'])
        )

    def to_dict(self):
        return dataclasses.asdict(self)

    def get_activation(self):
        return {
            'gelu': nn.GELU(),
        }[self.activation]

    def get_dtype(self):
        return {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
        }[self.dtype]

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
    truncate_steps: int
    "Max steps to run training (temporary)"
    clip_grad_norm: float
    "Norm for gradient clipping"
    min_p_halt: float
    "Minimum value for p_halt during training"
    ponder_target_loss: float
    "Percentile (from 0 to 1) to target as midpoint for pondering"
    optimizer: str
    "Optimizer to use"

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return cls(
            lr=float(obj['lr']),
            weight_decay=float(obj['weight_decay']),
            backspace_p=float(obj['backspace_p']),
            batch_size=int(obj['batch_size']),
            truncate_steps=int(obj['truncate_steps']),
            clip_grad_norm=float(obj['clip_grad_norm']),
            min_p_halt=float(obj['min_p_halt']),
            ponder_target_loss=float(obj['ponder_target_loss']),
            optimizer=str(obj['optimizer'])
        )

    def to_dict(self):
        return dataclasses.asdict(self)

    def make_optimizer(self, parameters):
        if self.optimizer == 'AdamW':
            return torch.optim.AdamW(
                parameters,
                self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(
                parameters,
                self.lr,
                weight_decay=self.weight_decay
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
            text = obj['text']
            if len(text) > 0:
                yield count, text
            count += 1

    if len(buffer) > 0:
        warnings.warn('dataset file did not end with newline')

def random_token_not(total: int, not_token: int):
    "Generate a random token id that is not the provided token"
    while True:
        token = torch.randint(0, total, tuple()).item()
        if token != not_token:
            return token

def tokenize_input(
    sp: SentencePieceProcessor,
    ctx_len: int,
    sequence: str,
):
    pad_start = [sp['<pad>']] * (ctx_len - 1) + [sp['<s>']]
    last = [sp['</s>']]
    encoded = sp.encode(sequence, out_type=int)
    return pad_start + encoded + last

# why doesn't safetensors support loading metadata?
def safetensors_load_metadata(filename):
    with open(filename, 'rb') as f:
        meta_len_b = f.read(8)
        meta_len, = struct.unpack('<Q', meta_len_b)
        meta_dict = json.loads(f.read(meta_len))
        return meta_dict['__metadata__']

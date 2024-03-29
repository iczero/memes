import dataclasses
import json
import struct
from typing import Self

import torch
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
    # TODO: "internal_seq_len" is a terrible name and is literally not even consistent
    "Length of internal sequence (total, short_ctx + recurrent)"
    recurrent_seq_len: int = dataclasses.field(init=False)
    "Length of recurrent sequence"
    ff_dropout_p: float
    "Probability of dropout after feedforward"
    attn_dropout_p: float
    "Probability of dropout after attention"
    n_intermediate: int
    "Number of intermediate layers"
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
    ponder_adjust: float
    "Offset for p_halt from center"

    def __post_init__(self):
        assert self.n_embed > 0
        assert self.n_attention_heads > 0
        assert self.vocab_size > 0
        assert self.short_ctx_len > 0
        assert self.internal_seq_len > 0
        assert self.internal_seq_len > self.short_ctx_len
        self.recurrent_seq_len = self.internal_seq_len - self.short_ctx_len
        assert self.ff_dropout_p >= 0
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
            resid_gate_multiplier=float(obj['resid_gate_multiplier']),
            activation=str(obj['activation']),
            dtype=str(obj['dtype']),
            qkv_bias=bool(obj['qkv_bias']),
            tokenizer_model_path=str(obj['tokenizer_model_path']),
            ponder_adjust=float(obj['ponder_adjust']),
        )

    def to_dict(self):
        return dataclasses.asdict(self)

    def get_activation(self):
        if self.activation == 'gelu':
            return nn.GELU()
        if self.activation == 'relu':
            return nn.ReLU()
        if self.activation == 'leakyrelu':
            return nn.LeakyReLU()

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
    "Max sequence length during training. Truncated BPTT didn't work and was removed"
    accumulate_gradients: int
    "How many batches to run before running the optimizer step"
    clip_grad_norm: float
    "Norm for gradient clipping"
    optimizer: str
    "Optimizer to use"
    min_p_halt: float
    "Lower bound for p_halt during training"
    confidence_scale: float
    "Scale factor for confidence loss when computing total loss"
    prev_loss_mean: float
    "Running mean of loss, for use in confidence"
    prev_loss_std: float
    "Running standard deviation of loss, for use in confidence"

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return cls(
            lr=float(obj['lr']),
            weight_decay=float(obj['weight_decay']),
            backspace_p=float(obj['backspace_p']),
            batch_size=int(obj['batch_size']),
            truncate_steps=int(obj['truncate_steps']),
            clip_grad_norm=float(obj['clip_grad_norm']),
            optimizer=str(obj['optimizer']),
            min_p_halt=float(obj['min_p_halt']),
            confidence_scale=float(obj['confidence_scale']),
            prev_loss_mean=float(obj['prev_loss_mean']),
            prev_loss_std=float(obj['prev_loss_std']),
            accumulate_gradients=int(obj['accumulate_gradients']),
        )

    def to_dict(self):
        return dataclasses.asdict(self)

    def make_optimizer(self, parameters):
        if self.optimizer == 'AdamW':
            return torch.optim.AdamW(
                parameters,
                self.lr,
                # test
                betas=(0.8, 0.95),
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(
                parameters,
                self.lr,
                weight_decay=self.weight_decay,
            )

def random_token_not(total: int, not_token: int):
    "Generate a random token id that is not the provided token"
    while True:
        token = torch.randint(0, total, tuple()).item()
        if token != not_token:
            return token

# why doesn't safetensors support loading metadata?
def safetensors_load_metadata(filename):
    with open(filename, 'rb') as f:
        meta_len_b = f.read(8)
        meta_len, = struct.unpack('<Q', meta_len_b)
        meta_dict = json.loads(f.read(meta_len))
        return meta_dict['__metadata__']

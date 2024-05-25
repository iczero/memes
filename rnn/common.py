import dataclasses
import enum
import json
import struct
from typing import Self

import torch
from torch import nn

@enum.verify(enum.CONTINUOUS)
class ControlTokens(enum.IntEnum):
    # 0 to 255 are for byte values
    PAD = 256
    "Padding token used for input"
    EMPTY = 257
    "Token used to initialize output positions"
    START_OF_TEXT = 258
    "Start of text token"
    END_OF_TEXT = 259
    "End of text token"

@dataclasses.dataclass
class ModelConfig:
    d_embed: int
    "Embedding dimensions"
    n_attention_heads: int
    "Number of attention heads"
    n_streams: int
    "Number of streams"
    # TODO: dropout is not currently implemented
    ff_dropout_p: float
    "Probability of dropout after feedforward"
    attn_dropout_p: float
    "Probability of dropout after attention"
    n_intermediate: int
    "Number of intermediate layers"
    resid_gate_multiplier: float
    "Multiplier for residual gating"
    d_ff_inner: int
    "Size of feedforward inner dimension (usually 4 * n_embed)"
    activation: str
    "Activation function"
    dtype: str
    "Data type of model"
    qkv_bias: bool
    "Whether Q/K/V linear layers in attention should have bias"
    short_ctx_len: int
    "Length of input short context"
    out_ctx_len: int
    "Length of output buffer"
    vocab_size: int = 256 + len(ControlTokens)
    "Vocabulary size, not read from config"

    def __post_init__(self):
        assert self.d_embed > 0
        assert self.n_attention_heads > 0
        assert self.vocab_size > 0
        assert self.ff_dropout_p >= 0
        assert self.resid_gate_multiplier > 0

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return cls(
            d_embed=int(obj['d_embed']),
            n_attention_heads=int(obj['n_attention_heads']),
            n_streams=int(obj['n_streams']),
            ff_dropout_p=float(obj['ff_dropout_p']),
            attn_dropout_p=float(obj['attn_dropout_p']),
            n_intermediate=int(obj['n_intermediate']),
            resid_gate_multiplier=float(obj['resid_gate_multiplier']),
            d_ff_inner=int(obj['d_ff_inner']),
            activation=str(obj['activation']),
            dtype=str(obj['dtype']),
            qkv_bias=bool(obj['qkv_bias']),
            short_ctx_len=int(obj['short_ctx_len']),
            out_ctx_len=int(obj['out_ctx_len']),
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

        raise RuntimeError('unknown activation')

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
    batch_size: int
    "Batch size"
    truncate_steps: int
    "Max backpropagation sequence length during training"
    # max_seq_len: int
    # "Max sequence length during training"
    accumulate_gradients: int
    "How many batches to run before running the optimizer step"
    clip_grad_norm: float
    "Norm for gradient clipping"
    optimizer: str
    "Optimizer to use"
    short_ctx_dropout_p: float
    "Probability to drop an input token from the short context"
    drift_commit_p_scale: float
    "Some value that goes into calculation of probability of commit"
    drift_commit_p_min: float
    "Minimum commit probability"
    drift_sample_temperature: float
    "Sampling temperature for token selection during training"

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return cls(
            lr=float(obj['lr']),
            weight_decay=float(obj['weight_decay']),
            batch_size=int(obj['batch_size']),
            truncate_steps=int(obj['truncate_steps']),
            # max_seq_len=int(obj['max_seq_len']),
            clip_grad_norm=float(obj['clip_grad_norm']),
            optimizer=str(obj['optimizer']),
            accumulate_gradients=int(obj['accumulate_gradients']),
            short_ctx_dropout_p=float(obj['short_ctx_dropout_p']),
            drift_commit_p_scale=float(obj['drift_commit_p_scale']),
            drift_commit_p_min=float(obj['drift_commit_p_min']),
            drift_sample_temperature=float(obj['drift_sample_temperature']),
        )

    def to_dict(self):
        return dataclasses.asdict(self)

    def make_param_groups(self, named_parameters):
        exclude_wd = []
        default = []
        for name, param in named_parameters:
            if len(param.shape) < 2 or name.endswith('.recurrent_init') or name.endswith('.bias'):
                exclude_wd.append(param)
            else:
                default.append(param)

        return [
            { 'params': exclude_wd, 'weight_decay': 0.0 },
            { 'params': default },
        ]

    def make_optimizer(self, named_parameters, allow_fused=False):
        groups = self.make_param_groups(named_parameters)
        if self.optimizer == 'AdamW':
            return torch.optim.AdamW(
                groups,
                self.lr,
                # test
                #betas=(0.8, 0.95),
                weight_decay=self.weight_decay,
                fused=allow_fused,
            )

        if self.optimizer == 'SGD':
            return torch.optim.SGD(
                groups,
                self.lr,
                weight_decay=self.weight_decay,
            )

        if self.optimizer == 'NAdamW':
            return torch.optim.NAdam(
                groups,
                self.lr,
                weight_decay=self.weight_decay,
                decoupled_weight_decay=True,
            )

        raise RuntimeError('unknown optimizer ' + self.optimizer)

# why doesn't safetensors support loading metadata?
def safetensors_load_metadata(filename):
    with open(filename, 'rb') as f:
        meta_len_b = f.read(8)
        meta_len, = struct.unpack('<Q', meta_len_b)
        meta_dict = json.loads(f.read(meta_len))
        return meta_dict['__metadata__']

def dump_sequence(tokens: list[int] | torch.Tensor) -> str:
    out = bytearray()
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()

    for token in tokens:
        if token < 256:
            out += token.to_bytes(1)
        else:
            out += f'[{ControlTokens(token).name}]'.encode()

    return repr(out.decode('utf-8', errors='replace'))

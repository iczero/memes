import torch
from torch import nn

# m = q, k index
# d = q, k dimension
# i = arange(1, d // 2)
# t = base ** (-2 * i / d)

# (batch, n_heads, seq, qkv_dim)

BASE = 10000
ROPE_DTYPE = torch.float32

class RotaryEncoding(nn.Module):
    "Implements rotary positional encoding (RoPE)"

    def __init__(self, hidden_dim: int, seq_len: int):
        super().__init__()
        self.cos_cached, self.sin_cached = rope_make_cached(hidden_dim, seq_len)

    def forward(self, x):
        # ensure cached values are on the right device
        if self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
        if self.sin_cached.device != x.device:
            self.sin_cached = self.sin_cached.to(x.device)

        return rotary_encoding_forward(self.cos_cached, self.sin_cached, x)

def rope_make_cached(hidden_dim: int, seq_len: int):
    assert hidden_dim % 2 == 0
    # compute coefficients at full precision
    dim_idx = torch.arange(0, hidden_dim, 2, dtype=torch.float64)
    theta = BASE ** (-dim_idx / hidden_dim)
    base = theta.unsqueeze(0) * torch.arange(seq_len).type_as(theta).unsqueeze(-1)
    return base.cos().type(ROPE_DTYPE), base.sin().type(ROPE_DTYPE)

def rotary_encoding_forward(cos_cached: torch.Tensor, sin_cached: torch.Tensor, x: torch.Tensor):
    # [[ cos(m*t), -sin(m*t) ],  @  [[a],
    #  [ sin(m*t),  cos(m*t) ]]      [b]]
    # a2: cos(m*t) * a + sin(m*t) * -b
    # b2: sin(m*t) * a + cos(m*t) *  b

    # in: (batch..., seq, n_embed)
    # maintain full precision
    x_dt = x.type(ROPE_DTYPE)
    a = x_dt[..., 0::2]
    b = x_dt[..., 1::2]
    a2 = cos_cached * a + sin_cached * -b
    b2 = sin_cached * a + cos_cached * b
    return torch.stack((a2, b2), dim=-1).flatten(-2, -1).type_as(x)

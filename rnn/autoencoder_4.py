import math

import aim
import torch
import torch._dynamo.config
import torch.nn.functional as F
from torch import nn

from .common import ModelConfig, dump_sequence
from .data import filter_text, load_dataset

class GLU(nn.Module):
    def __init__(self, in_dim, out_dim, activation, bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.activation = activation
        self.w = nn.Linear(in_dim, out_dim * 2, bias=bias)

    def forward(self, x: torch.Tensor):
        w1, w2 = self.w(x).split(self.out_dim, dim=-1)
        return w1 * self.activation(w2)

class BatchLinear(nn.Module):
    def __init__(self, n_streams: int, d_in: int, d_out: int, has_bias = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.has_bias = has_bias

        self.weight = nn.Parameter(torch.zeros((n_streams, d_out, d_in)))
        if has_bias:
            self.bias = nn.Parameter(torch.zeros((n_streams, d_out)))

        self.reset_parameters()

    def reset_parameters(self):
        # uniform initialization between (-1/sqrt(in_features), 1/sqrt(in_features))
        bound = 1 / math.sqrt(self.d_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.has_bias:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        # in: (batch, stream, d_embed)
        # x @ W^T + b
        # transpose to (stream, batch, d_embed) then batch matmul and transpose back
        x = torch.bmm(x.transpose(-2, -3), self.weight.mT).transpose(-2, -3)
        if self.has_bias:
            # add bias (broadcasted): (batch, stream, d_embed) + (1, stream, d_embed)
            x = x + self.bias

        return x

class BatchGLU(nn.Module):
    def __init__(self, n_streams: int, d_in: int, d_mid: int, d_out: int, activation, has_bias = True):
        super().__init__()
        self.d_mid = d_mid
        self.w_in = BatchLinear(n_streams, d_in, d_mid * 2, has_bias=has_bias)
        self.activation = activation
        self.w_out = BatchLinear(n_streams, d_mid, d_out, has_bias=has_bias)

    def forward(self, x: torch.Tensor):
        mid_1, mid_2 = self.w_in(x).split(self.d_mid, dim=-1)
        # print('mid_1 max:', mid_1.abs().max())
        # print('mid_2 max:', mid_2.abs().max())
        mid = self.activation(mid_1) * mid_2
        return self.w_out(mid)

def apply_inline_gate(x: torch.Tensor, gate_multiplier: float):
    # (..., d_embed + 1) -> (..., d_embed)
    resid_gate = F.sigmoid(x[..., -1]) * gate_multiplier
    return x[..., :-1] * resid_gate.unsqueeze(-1)

class CrossAttentionInput(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_attention_heads = config.n_attention_heads
        self.resid_gate_multiplier = config.resid_gate_multiplier

        self.input_embedding = nn.Embedding(config.vocab_size, config.d_embed)
        # marker for uncommitted
        self.uncommitted_marker = nn.Parameter(torch.randn((config.d_embed,)))
        self.positional_encodings = nn.Parameter(
            torch.randn((config.short_ctx_len + config.out_ctx_len, config.d_embed))
        )
        self.recurrent_norm = nn.LayerNorm([config.d_embed])
        self.q_linear = BatchLinear(
            config.n_streams,
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            has_bias=config.qkv_bias,
        )
        self.kv_linear = nn.Linear(
            config.d_embed,
            2 * config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads + 1,)))

        ff_in_dim = config.d_embed * (config.n_attention_heads + 1)
        self.feedforward = BatchGLU(
            n_streams=config.n_streams,
            d_in=ff_in_dim,
            d_mid=config.d_ff_inner,
            d_out=config.d_embed + 1,
            activation=config.get_activation(),
        )

    def forward(self, recurrent: torch.Tensor, inputs: torch.Tensor, committed: torch.Tensor):
        recurrent_norm = self.recurrent_norm(recurrent)
        # (batch, stream) -> (batch, stream, d_embed)
        inputs_embed = self.input_embedding(inputs)
        # mark uncommitted positions
        inputs_embed = inputs_embed + torch.where(committed.unsqueeze(-1), 0, self.uncommitted_marker)
        # apply positional encoding to byte embeddings directly
        inputs_embed = inputs_embed + self.positional_encodings

        q = self.q_linear(recurrent_norm) \
            .unflatten(-1, (self.n_attention_heads, self.d_embed)) \
            .transpose(-2, -3)
        # kv_merged: (batch, stream, 2 (k/v), n_heads, d_embed)
        kv_merged = self.kv_linear(inputs_embed) \
            .unflatten(-1, (2, self.n_attention_heads, self.d_embed))
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # -> (batch, stream, n_heads, d_embed)
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)
        # add skip
        attn_out = torch.cat((attn_out, recurrent_norm.unsqueeze(-2)), dim=-2)
        # scale heads
        attn_out = self.head_scales.unsqueeze(-1) * attn_out

        attn_concat = attn_out.flatten(-2, -1)
        # attn_concat: (batch, stream, n_heads * d_embed + 1)
        ff_out = self.feedforward(attn_concat)

        resid = apply_inline_gate(ff_out, self.resid_gate_multiplier)
        return resid

class Intermediate(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_attention_heads = config.n_attention_heads
        self.resid_gate_multiplier = config.resid_gate_multiplier

        self.norm = nn.LayerNorm([config.d_embed])
        self.q_linear = BatchLinear(
            config.n_streams,
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            has_bias=config.qkv_bias,
        )
        self.kv_linear = nn.Linear(
            config.d_embed,
            2 * config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads + 1,)))

        ff_in_dim = config.d_embed * (config.n_attention_heads + 1)
        self.feedforward = BatchGLU(
            n_streams=config.n_streams,
            d_in=ff_in_dim,
            d_mid=config.d_ff_inner,
            d_out=config.d_embed + 1,
            activation=config.get_activation(),
        )

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        q = self.q_linear(x_norm) \
            .unflatten(-1, (self.n_attention_heads, self.d_embed)) \
            .transpose(-2, -3)
        kv_merged = self.kv_linear(x_norm) \
            .unflatten(-1, (2, self.n_attention_heads, self.d_embed))
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # -> (batch, stream, n_heads, d_embed)
        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)
        # add skip
        attn_out = torch.cat((attn_out, x_norm.unsqueeze(-2)), dim=-2)
        # scale heads
        attn_out = self.head_scales.unsqueeze(-1) * attn_out

        attn_concat = attn_out.flatten(-2, -1)
        # attn_concat: (batch, stream, n_heads * d_embed + 1)
        ff_out = self.feedforward(attn_concat)

        resid = apply_inline_gate(ff_out, self.resid_gate_multiplier)
        return resid

class PreOutput(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_attention_heads = config.n_attention_heads

        self.norm = nn.LayerNorm([config.d_embed])
        self.q_linear = BatchLinear(
            config.n_streams,
            config.d_embed,
            config.n_attention_heads * config.d_embed,
            has_bias=config.qkv_bias,
        )
        self.kv_linear = nn.Linear(
            config.d_embed,
            2 * config.n_attention_heads * config.d_embed,
            bias=config.qkv_bias,
        )
        self.head_scales = nn.Parameter(torch.ones((config.n_attention_heads,)))

        # no skip
        ff_in_dim = config.d_embed * config.n_attention_heads
        self.feedforward = BatchGLU(
            n_streams=config.n_streams,
            d_in=ff_in_dim,
            d_mid=config.d_ff_inner,
            d_out=config.d_embed,
            activation=config.get_activation(),
        )

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        q = self.q_linear(x_norm) \
            .unflatten(-1, (self.n_attention_heads, self.d_embed)) \
            .transpose(-2, -3)
        kv_merged = self.kv_linear(x_norm) \
            .unflatten(-1, (2, self.n_attention_heads, self.d_embed))
        k = kv_merged[..., 0, :, :].transpose(-2, -3)
        v = kv_merged[..., 1, :, :].transpose(-2, -3)

        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)
        # scale heads
        attn_out = self.head_scales.unsqueeze(-1) * attn_out
        attn_concat = attn_out.flatten(-2, -1)
        ff_out = self.feedforward(attn_concat)
        return ff_out

class CharDecode(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.out_query = nn.Parameter(torch.randn((config.out_ctx_len, config.d_embed,)))
        self.kv_linear = nn.Linear(config.d_embed, config.d_embed * 2, bias=config.qkv_bias)

        self.feedforward = GLU(config.d_embed, config.d_embed, config.get_activation())
        self.w_out = nn.Linear(config.d_embed, config.vocab_size)

    def forward(self, x: torch.Tensor):
        kv_merged = self.kv_linear(x).unflatten(-1, (2, self.d_embed))
        # kv_merged: (batch, stream, 2, n_embed)
        # no transpose since we have no n_heads
        k = kv_merged[..., 0, :]
        v = kv_merged[..., 1, :]
        q = self.out_query

        # pylint: disable-next=not-callable
        attn_out = F.scaled_dot_product_attention(q, k, v)
        embeddings_out = self.feedforward(attn_out)
        logits_out = self.w_out(embeddings_out)
        return embeddings_out, logits_out

class Autoencoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.recurrent_init = nn.Parameter(torch.zeros(config.n_streams, config.d_embed))
        self.input0 = CrossAttentionInput(config)
        self.int1 = Intermediate(config)
        self.int2 = Intermediate(config)
        self.output3 = PreOutput(config)
        self.output4 = PreOutput(config)
        self.char_decode = CharDecode(config)

        init_bound = 1 / math.sqrt(config.d_embed)
        nn.init.uniform_(self.recurrent_init, -init_bound, init_bound)

    def forward(
        self,
        recurrent: torch.Tensor,
        inputs: torch.Tensor,
        inputs_committed: torch.Tensor,
    ):
        recurrent = recurrent + self.input0(recurrent, inputs, inputs_committed)
        recurrent = recurrent + self.int1(recurrent)
        recurrent = recurrent + self.int2(recurrent)
        recurrent = self.output3(recurrent)
        recurrent = self.output4(recurrent)

        embeddings_out, logits_out = self.char_decode(recurrent)

        return recurrent, embeddings_out, logits_out

def to_bytes_list(s: str) -> list[int]:
    return list(s.encode('utf-8'))

def make_param_groups(named_parameters):
    exclude_wd = []
    default = []
    for name, param in named_parameters:
        if len(param.shape) < 2 or name.endswith('.recurrent_init') \
                or name.endswith('.bias') or name.endswith('.out_query') \
                or name.endswith('.positional_encodings'):
            exclude_wd.append(param)
        else:
            default.append(param)

    return [
        { 'params': exclude_wd, 'weight_decay': 0.0 },
        { 'params': default },
    ]

def main():
    run = aim.Run()
    run.experiment = 'autoencoder-test'
    run.name = 'autoencoder-4.4'

    model_config = ModelConfig(
        d_embed=128,
        n_attention_heads=6,
        n_streams=32,
        ff_dropout_p=0.0,
        attn_dropout_p=0.0,
        n_intermediate=0,
        resid_gate_multiplier=1.0,
        d_ff_inner=512,
        activation='gelu',
        dtype='float32',
        qkv_bias=True,
        short_ctx_len=0,
        out_ctx_len=128,
    )
    batch_size = 128

    data_path = '/mnt/data/opt/the-pile/train/00.jsonl.zst'
    data_iter = filter_text(load_dataset(open(data_path, 'rb')))
    def slice_stuff(data, target_len: int):
        for el in data:
            bytes_list = to_bytes_list(el)
            if len(bytes_list) < target_len:
                continue

            yield bytes_list[:target_len]

    data_iter = slice_stuff(data_iter, model_config.out_ctx_len)

    device = torch.device('cuda')
    model = Autoencoder(model_config)
    model.to(device=device, dtype=model_config.get_dtype())

    optimizer = torch.optim.AdamW(
        make_param_groups(model.named_parameters()),
        lr=1e-4,
        weight_decay=0.05
    )

    committed_mask = torch.full((batch_size, model_config.out_ctx_len), True, device=device)
    step = 0
    while True:
        step += 1
        print('step:', step)
        sequences = torch.tensor([next(data_iter) for n in range(batch_size)])
        sequences = sequences.to(device)

        optimizer.zero_grad()
        _, _, output = model(model.recurrent_init.unsqueeze(0) \
            .repeat_interleave(batch_size, dim=0), sequences, committed_mask)
        loss = F.cross_entropy(output.transpose(-2, -1), sequences, reduction='mean')
        loss.backward()
        grad_norm_f = nn.utils.clip_grad_norm_(
            model.parameters(),
            30.0,
            error_if_nonfinite=True,
        ).item()
        optimizer.step()

        run.track(loss.item(), name='loss', step=step)
        run.track(grad_norm_f, name='grad_norm', step=step)

        if step % 25 == 0:
            for i in range(10):
                seq_in = dump_sequence(sequences[i])
                seq_out = dump_sequence(output[i].argmax(dim=-1))

                text1 = f'batch {i}'
                print(f'{text1} in  {seq_in}')
                print(f'{" " * len(text1)} out {seq_out}')

if __name__ == '__main__':
    main()

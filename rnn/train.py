from collections.abc import Iterable
import dataclasses
import sys
from pathlib import Path

import aim
import safetensors
import sentencepiece as spm
import torch
import torch.nn.functional as F
import tqdm

from .common import ModelConfig, TrainConfig, load_dataset, tokenize_input
from .model import RNNSequence

@torch.jit.script
def ponder_loss(
    loss: torch.Tensor, p_halt: torch.Tensor,
    continue_penalty: float, loss_penalty: float
):
    return continue_penalty * (1 - p_halt) + loss * loss_penalty ** p_halt

@dataclasses.dataclass
class TrainSequence:
    sequence: list[int]
    offset: int = 0
    ended: bool = False
    "List of losses per step"
    losses: list[torch.Tensor] = []
    "List of halting probability per ponder step"
    p_halt: list[float] = []
    "Running probability that all previous ponder steps have not halted"
    p_not_halt: torch.Tensor = torch.tensor(1.)
    "Previous internal state, if not halted"
    prev_internal: torch.Tensor | None

class Trainer:
    model_config: ModelConfig
    train_config: TrainConfig
    device: torch.device
    dtype: torch.dtype
    sequences: list[TrainSequence] = []
    model: RNNSequence
    recurrent: torch.Tensor
    tokenizer: spm.SentencePieceProcessor
    train_iter: Iterable[str]
    "Previous batched internal state, if reusable"
    prev_internal: torch.Tensor | None
    "List of sequences by index which halted the last step"
    halted_sequences: list[int]

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        tokenizer: spm.SentencePieceProcessor,
        train_set: Path,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.model = RNNSequence(model_config)
        self.model.type(self.dtype)
        self.model.to(self.device)

        self.train_file = open(train_set, 'rb')
        self.train_iter = load_dataset(self.train_file)

    def next_sequence(self) -> list[int]:
        # TODO: what if this implodes
        seq = next(self.train_iter)
        return tokenize_input(
            self.tokenizer,
            self.model_config.short_ctx_len,
            seq,
            train=True,
            backspace_p=self.train_config.backspace_p
        )

    def reset(self):
        batch_size = self.train_config.batch_size
        self.recurrent = self.model.recurrent_init \
            .unsqueeze(0).repeat_interleave(batch_size, 0)
        self.sequences = [
            TrainSequence(self.next_sequence()) for i in range(batch_size)
        ]
        self.prev_internal = None
        self.halted_sequences = list(range(batch_size))

    def prepare_internal_batch(self):
        short_ctx_len = self.model_config.short_ctx_len
        if len(self.halted_sequences) == 0 and self.prev_internal is not None:
            # no previous steps have halted, can reuse internal
            return self.prev_internal
        else:
            to_encode: list[tuple[int, torch.Tensor]] = []
            internal_batch: list[torch.Tensor | None] = []
            for i, info in enumerate(self.sequences):
                if info.prev_internal is not None:
                    assert i not in self.halted_sequences
                    internal_batch.append(info.prev_internal)
                else:
                    assert i in self.halted_sequences
                    assert not info.ended
                    # prepare batch for input layer
                    short_ctx = info.sequence[info.offset : info.offset + short_ctx_len]
                    short_ctx = torch.tensor(short_ctx, dtype=self.dtype, device=self.device)
                    to_encode.append((i, short_ctx))
                    # will be substitued later
                    internal_batch.append(None)

            if len(to_encode) > 0:
                # run input batch
                input_encode = torch.stack([v[1] for v in to_encode], dim=0)
                input_encode = self.model.input(input_encode)
                # input_encode first dimension is batch
                for item, encoded in zip(to_encode, input_encode):
                    internal_batch[item[0]] = encoded

            return torch.stack(internal_batch, dim=0)

    def forward_step(self):
        internal = self.prepare_internal_batch()
        next_recurrent, next_internal = self.model.ponder(self.recurrent, internal)
        self.recurrent = next_recurrent
        token_out, p_halt_out = self.model.decode(next_recurrent)
        halted = p_halt_out.bernoulli().nonzero().flatten().tolist()

        # collect expected tokens
        expected_tokens = [info.sequence[info.offset] for info in self.sequences]
        expected_tokens = torch.tensor(expected_tokens, device=self.device)
        cross_entropy = F.cross_entropy(token_out, expected_tokens)
        step_loss = ponder_loss(
            cross_entropy,
            p_halt_out,
            self.model_config.ponder_continue_penalty,
            self.model_config.ponder_loss_penalty
        )

        has_any_halt = False
        for i, info in enumerate(self.sequences):
            pass # TODO:

def main():
    in_path = Path(sys.argv[1])
    validation_set = in_path / 'val.jsonl.zst'
    train_set = in_path / 'train' / '00.jsonl.zst'
    device = torch.device('cuda')
    dtype = torch.float32

if __name__ == '__main__':
    main()

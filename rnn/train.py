import dataclasses
import sys
from collections.abc import Iterable
from pathlib import Path

import aim
import torch
import torch.nn.functional as F
import tqdm
from safetensors.torch import save_model, load_model
from sentencepiece import SentencePieceProcessor
from torch import nn

from .common import ModelConfig, TrainConfig, load_dataset, tokenize_input
from .model import RNNSequence


def dprint(*args):
    return
    #print('debug:', *args)

@torch.jit.script
def ponder_loss(
    loss: torch.Tensor, p_halt: torch.Tensor,
    continue_penalty: float, loss_penalty: float
):
    return continue_penalty * (1 - p_halt) + loss * loss_penalty ** p_halt

@dataclasses.dataclass
class TrainSequence:
    sequence: list[int]
    "Token sequence"
    p_not_halt: torch.Tensor
    "Running probability that all previous ponder steps have not halted"
    offset: int = 0
    "Current offset into the seqeuence"
    ended: bool = False
    "If the sequence has reached the end"
    losses: list[torch.Tensor] = dataclasses.field(default_factory=list)
    "List of losses for all steps"
    halted_losses: list[torch.Tensor] = dataclasses.field(default_factory=list)
    "Unweighted cross entropy loss at the end of each halted step"
    prev_internal: torch.Tensor | None = None
    "Previous internal state, if not halted"
    total_ponder: int = 0
    "Total number of steps repeated for ponder at this offset"

class Trainer:
    model_config: ModelConfig
    train_config: TrainConfig
    device: torch.device
    dtype: torch.dtype
    sequences: list[TrainSequence] = []
    model: RNNSequence
    recurrent: torch.Tensor
    tokenizer: SentencePieceProcessor
    train_iter: Iterable[str]
    prev_internal: torch.Tensor | None
    "Previous batched internal state, if reusable"
    halted_sequences: list[int]
    "List of sequences by index which halted the last step"

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        tokenizer: SentencePieceProcessor,
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
        self.model.train()

        self.train_file = open(train_set, 'rb')
        self.train_iter = load_dataset(self.train_file)

    def next_sequence(self) -> list[int]:
        # TODO: what if this implodes
        count, seq = next(self.train_iter)
        return tokenize_input(
            self.tokenizer,
            self.model_config.short_ctx_len,
            seq,
            train=True,
            backspace_p=self.train_config.backspace_p
        )

    def next_batch(self):
        batch_size = self.train_config.batch_size
        self.recurrent = self.model.recurrent_init \
            .unsqueeze(0).repeat_interleave(batch_size, 0)
        self.sequences = [
            TrainSequence(
                sequence=self.next_sequence(),
                p_not_halt=torch.tensor(1., dtype=self.dtype, device=self.device),
            ) for _ in range(batch_size)
        ]
        self.prev_internal = None
        self.halted_sequences = list(range(batch_size))

    def prepare_internal_batch(self):
        short_ctx_len = self.model_config.short_ctx_len
        if len(self.halted_sequences) == 0 and self.prev_internal is not None:
            # no previous steps have halted, can reuse internal
            dprint('prepare_internal_batch: will reuse prev_internal')
            return self.prev_internal
        else:
            # construct batch for input layer
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
                    short_ctx = torch.tensor(short_ctx, dtype=torch.int32, device=self.device)
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
        should_halt = p_halt_out.bernoulli().nonzero().flatten().tolist()

        # collect expected tokens
        expected_tokens = [
            # next token
            info.sequence[info.offset + self.model_config.short_ctx_len]
            for info in self.sequences
        ]
        expected_tokens = torch.tensor(expected_tokens, device=self.device)
        cross_entropy = F.cross_entropy(token_out, expected_tokens, reduction='none')
        step_losses = ponder_loss(
            cross_entropy,
            p_halt_out,
            self.model_config.ponder_continue_penalty,
            self.model_config.ponder_loss_penalty
        )

        self.halted_sequences.clear()
        has_halt = False
        all_ended = True
        for i, info in enumerate(self.sequences):
            if info.ended:
                # skip finished sequences
                continue

            all_ended = False
            p_halt_detached = p_halt_out[i].clone().detach()
            did_halt = i in should_halt
            if not did_halt and info.total_ponder >= self.train_config.max_ponder_steps:
                # force halt
                p_halt_detached.copy_(1.)
                did_halt = True

            # P(halt | not previously halted) * ponder step loss
            weighted_loss = info.p_not_halt * p_halt_detached * step_losses[i]
            info.losses.append(weighted_loss)

            if did_halt:
                info.prev_internal = None
                info.total_ponder = 0
                info.p_not_halt.copy_(1.)
                # record unweighted loss as well
                info.halted_losses.append(cross_entropy[i])

                # check if sequence ended
                # we end one token before the last otherwise there is no "next" token to train on
                if info.offset + self.model_config.short_ctx_len >= len(info.sequence) - 1:
                    info.ended = True
                    # introduce dummy internal sequence
                    info.prev_internal = torch.zeros((
                        self.model_config.short_ctx_len,
                        self.model_config.n_embed,
                    ), device=self.device, dtype=self.dtype)
                else:
                    # slide shortctx to include next token
                    info.offset += 1
                    self.halted_sequences.append(i)
                    has_halt = True
            else:
                info.prev_internal = next_internal[i]
                info.p_not_halt *= 1 - p_halt_detached
                info.total_ponder += 1

        # stash internal state if we can reuse it
        self.prev_internal = next_internal if not has_halt else None

        return all_ended

    def sum_train_loss(self) -> torch.Tensor:
        sequence_losses = []
        for info in self.sequences:
            sequence_losses.append(torch.stack(info.losses).sum() / info.offset)

        return torch.stack(sequence_losses).sum() / len(self.sequences)

    def sum_validation_loss(self) -> torch.Tensor:
        sequence_losses = []
        for info in self.sequences:
            sequence_losses.append(torch.stack(info.halted_losses).sum() / info.offset)

        return torch.stack(sequence_losses).sum() / len(self.sequences)

def main():
    in_path = Path(sys.argv[1])
    validation_set = in_path / 'val.jsonl.zst'
    train_set = in_path / 'train' / '00.jsonl.zst'
    device = torch.device('cuda')
    dtype = torch.float32

    #torch.autograd.set_detect_anomaly(True)

    model_config = ModelConfig.default()
    train_config = TrainConfig.default()
    tokenizer = SentencePieceProcessor()
    tokenizer.Init(model_file='data/tokenizer6.model')

    trainer = Trainer(model_config, train_config, tokenizer, train_set, device, dtype)

    model = trainer.model
    #load_model(model, 'rnn.model')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        train_config.lr,
        weight_decay=train_config.weight_decay
    )
    batch = 0
    while True:
        batch += 1
        optimizer.zero_grad()

        trainer.next_batch()
        done = False
        steps = 0
        while not done:
            steps += 1
            done = trainer.forward_step()
            if steps >= train_config.max_steps_temp:
                break

        loss = trainer.sum_train_loss()
        show_loss = loss.item()
        show_unweighted_loss = trainer.sum_validation_loss().item()
        loss.backward()
        show_grads_norm = nn.utils.clip_grad_norm_(
            model.parameters(), 3., error_if_nonfinite=True
        ).item()
        optimizer.step()

        print('\nbatch:', batch)
        print('grad norm:', show_grads_norm)
        print('training loss:', show_loss)
        print('unweighted loss:', show_unweighted_loss)

        if batch % 500 == 0:
            save_model(model, 'rnn-test.model')

if __name__ == '__main__':
    main()

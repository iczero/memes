import dataclasses
import sys
from collections.abc import Iterable
from pathlib import Path

import aim
import torch
import torch.nn.functional as F
from safetensors.torch import load_model, save_model
from sentencepiece import SentencePieceProcessor
from torch import nn

from .common import (ModelConfig, TrainConfig, load_dataset, random_token_not,
                     tokenize_input)
from .model import RNNSequence

DISABLE_TORCH_COMPILE = False

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
    was_backspace: bool = False
    "If the last token was randomized for backspace"

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
        return tokenize_input(self.tokenizer, self.model_config.short_ctx_len, seq)

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
        self.halted_sequences = list(range(batch_size))

    @torch.compile(disable=DISABLE_TORCH_COMPILE)
    def forward_input_batch(self, input_encode: torch.Tensor):
        return self.model.input(input_encode)

    def prepare_internal_batch(self):
        short_ctx_len = self.model_config.short_ctx_len
        # construct batch for input layer
        input_batch: list[tuple[int, torch.Tensor]] = []
        input_sequences: list[torch.Tensor | None] = []
        output_list: list[torch.Tensor] = []
        for i, info in enumerate(self.sequences):
            if info.prev_internal is not None:
                assert i not in self.halted_sequences
                input_sequences.append(info.prev_internal)
            else:
                assert i in self.halted_sequences
                assert not info.ended
                # prepare batch for input layer
                short_ctx = info.sequence[info.offset : info.offset + short_ctx_len]
                if info.offset > 4 and torch.bernoulli(
                    torch.tensor(self.train_config.backspace_p)
                ).item() > 0:
                    short_ctx[-1] = random_token_not(len(self.tokenizer), short_ctx[-1])
                    info.offset -= 1
                    info.was_backspace = True
                short_ctx = torch.tensor(short_ctx, dtype=torch.int64, device=self.device)
                input_batch.append((i, short_ctx))
                # will be substitued later
                input_sequences.append(None)

            # grab next token
            if info.was_backspace:
                next_token = self.tokenizer['<del>']
            else:
                next_token = info.sequence[info.offset + short_ctx_len]
            next_token = torch.tensor(next_token, dtype=torch.int64, device=self.device)
            output_list.append(next_token)

        if len(input_batch) > 0:
            # run input batch
            input_encode = torch.stack([v[1] for v in input_batch], dim=0)
            input_encode = self.forward_input_batch(input_encode)

            # _tmp_idx = 0
            for item, encoded in zip(input_batch, input_encode):
                # input_encode first dimension is batch
                input_sequences[item[0]] = encoded

                # print('\nprepare_internal_batch(): sequences dump')
                # print(
                #     '  batch element:', item[0],
                #     repr(''.join(self.tokenizer.IdToPiece(p) for p in input_batch[_tmp_idx][1].tolist())),
                #     '->',
                #     repr(self.tokenizer.IdToPiece(output_list[item[0]].item())),
                # )
                # _tmp_idx += 1

        input_array = torch.stack(input_sequences, dim=0)
        output_array = torch.stack(output_list, dim=0)

        return input_array, output_array

    @torch.compile(disable=DISABLE_TORCH_COMPILE)
    def forward_ponder_batch(
        self,
        recurrent: torch.Tensor,
        internal: torch.Tensor,
        expected_tokens: torch.Tensor
    ):
        next_recurrent, next_internal = self.model.ponder(recurrent, internal)
        token_out, p_halt_out = self.model.decode(next_recurrent, next_internal)

        cross_entropy = F.cross_entropy(token_out, expected_tokens, reduction='none')
        step_losses = ponder_loss(
            cross_entropy,
            p_halt_out,
            self.model_config.ponder_continue_penalty,
            self.model_config.ponder_loss_penalty
        )

        with torch.no_grad():
            should_halt = p_halt_out.bernoulli()

        return next_recurrent, next_internal, token_out, p_halt_out, \
            should_halt, cross_entropy, step_losses

    def forward_step(self):
        internal, expected_tokens = self.prepare_internal_batch()

        next_recurrent, next_internal, _token_out, p_halt_out, should_halt, \
            cross_entropy, step_losses = \
            self.forward_ponder_batch(self.recurrent, internal, expected_tokens)

        # print('forward_step(): p_halt', p_halt_out)

        should_halt = should_halt.nonzero().flatten().tolist()
        self.recurrent = next_recurrent
        self.halted_sequences.clear()
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
                    if not info.was_backspace:
                        info.offset += 1
                    else:
                        info.was_backspace = False
                    self.halted_sequences.append(i)
            else:
                info.prev_internal = next_internal[i]
                info.p_not_halt *= 1 - p_halt_detached
                info.total_ponder += 1

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
    #validation_set = in_path / 'val.jsonl.zst'
    train_set = in_path / 'train' / '01.jsonl.zst'
    device = torch.device('cuda')
    dtype = torch.float32

    #torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('high')

    model_config = ModelConfig.default()
    train_config = TrainConfig.default()
    tokenizer = SentencePieceProcessor()
    tokenizer.Init(model_file='data/tokenizer7.model')

    trainer = Trainer(model_config, train_config, tokenizer, train_set, device, dtype)

    model = trainer.model
    try:
        # TODO: save hyperparameters and training progress to the checkpoint
        load_model(model, 'rnn-load.model')
    except FileNotFoundError:
        print('warning: could not find file to load')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        train_config.lr,
        weight_decay=train_config.weight_decay
    )

    run = aim.Run()
    run['model_config'] = dataclasses.asdict(model_config)
    run['train_config'] = dataclasses.asdict(train_config)

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
        try:
            show_grads_norm = nn.utils.clip_grad_norm_(
                model.parameters(), 3., error_if_nonfinite=True
            ).item()
        except RuntimeError as e:
            print('\nwarning: clip_grad_norm_ failed:', e)
            print('  the current step will be skipped')
            print('batch:', batch)
            print('loss:', show_loss)
            continue

        optimizer.step()
        print('\nbatch:', batch)
        print('grad norm:', show_grads_norm)
        print('training loss:', show_loss)
        print('unweighted loss:', show_unweighted_loss)
        run.track(show_loss, name='loss', step=batch)
        run.track(show_grads_norm, name='grad_norm', step=batch)

        if batch % 500 == 0:
            save_model(model, 'rnn-test.model')

if __name__ == '__main__':
    main()

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


DISABLE_TORCH_COMPILE = False

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

    @torch.compile(disable=DISABLE_TORCH_COMPILE)
    def forward_input_batch(self, input_encode: torch.Tensor):
        return self.model.input(input_encode)

    def prepare_internal_batch(self):
        short_ctx_len = self.model_config.short_ctx_len
        input_list = []
        output_list = []
        for info in self.sequences:
            # prepare batch for input layer
            short_ctx = info.sequence[info.offset : info.offset + short_ctx_len]
            short_ctx = torch.tensor(short_ctx, dtype=torch.int64, device=self.device)
            input_list.append(short_ctx)
            next_token = info.sequence[info.offset + short_ctx_len]
            next_token = torch.tensor(next_token, dtype=torch.int64, device=self.device)
            output_list.append(next_token)

        input_array = torch.stack(input_list, dim=0)
        input_array = self.forward_input_batch(input_array)
        output_array = torch.stack(output_list, dim=0)

        # print('prepare_internal_batch(): sequences dump')
        # for t_in, t_out in zip(input_list, output_list):
        #     t_in = t_in.to('cpu').tolist()
        #     t_out = t_out.item()
        #     print(
        #         repr(''.join(self.tokenizer.IdToPiece(p) for p in t_in)),
        #         '->',
        #         repr(self.tokenizer.IdToPiece(t_out))
        #     )

        return input_array, output_array

    @torch.compile(disable=DISABLE_TORCH_COMPILE)
    def forward_ponder_batch(
        self,
        recurrent: torch.Tensor,
        internal: torch.Tensor,
        expected_tokens: torch.Tensor
    ):
        next_recurrent, next_internal = self.model.ponder(recurrent, internal)
        token_out = self.model.decode(next_recurrent, next_internal)

        cross_entropy = F.cross_entropy(token_out, expected_tokens, reduction='none')

        return next_recurrent, next_internal, token_out, None, \
            None, cross_entropy, None

    def forward_step(self):
        internal, expected_tokens = self.prepare_internal_batch()

        next_recurrent, _next_internal, _token_out, _, _, \
            cross_entropy, _ = \
            self.forward_ponder_batch(self.recurrent, internal, expected_tokens)

        self.recurrent = next_recurrent
        all_ended = True
        for i, info in enumerate(self.sequences):
            if info.ended:
                # skip finished sequences
                continue

            all_ended = False

            # check if sequence ended
            # we end one token before the last otherwise there is no "next" token to train on
            if info.offset + self.model_config.short_ctx_len >= len(info.sequence) - 1:
                info.ended = True
            else:
                info.losses.append(cross_entropy[i])
                # slide shortctx to include next token
                info.offset += 1

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
    train_set = in_path / 'train' / '00.jsonl.zst'
    device = torch.device('cuda')
    dtype = torch.float32

    #torch.autograd.set_detect_anomaly(True)
    #torch.set_float32_matmul_precision('high')

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
        #show_unweighted_loss = trainer.sum_validation_loss().item()
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
        #print('unweighted loss:', show_unweighted_loss)

        #if batch == 50:
        #    import IPython
        #    IPython.embed()

        if batch % 500 == 0:
            save_model(model, 'rnn-test.model')

        if batch >= 30_000:
            return

if __name__ == '__main__':
    main()

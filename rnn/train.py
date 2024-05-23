import dataclasses
import datetime
import json
import signal
import sys

import aim
import torch
import torch._dynamo.config
import torch.nn.functional as F
from torch import nn

from .common import ControlTokens, ModelConfig, TrainConfig
from .data import SequenceProvider, filter_text, load_dataset
from .model import RNN

DISABLE_TORCH_COMPILE = False
"If torch.compile should be disabled"

# extreme hack (temporarily permanent)
DEBUG_RECURRENT_GRAD: list[torch.Tensor] | bool | None = None
def signal_grad_debug(*_args):
    global DEBUG_RECURRENT_GRAD
    DEBUG_RECURRENT_GRAD = True
signal.signal(signal.SIGUSR1, signal_grad_debug)

def batch_roll(data: torch.Tensor, shift_by: torch.Tensor) -> torch.Tensor:
    """
    Batched `torch.roll()`

    `batch`: (batch, stream, d_embed) \\
    `shift_by`: (batch), torch.int64 \\
    returns `batch` rolled "left" by `shift_by
    """

    roll_size = data.shape[-2]
    # generate indices
    index_range = torch.arange(roll_size)
    # shift indices by shift_by, wrap around
    shift_mask = (index_range.unsqueeze(0) + shift_by.unsqueeze(-1)) % roll_size
    # broadcast shift_mask to fit data
    shift_mask = shift_mask.unsqueeze(-1).expand(data.shape)
    # do the roll
    return torch.gather(data, -2, shift_mask)

@dataclasses.dataclass
class TrainSequence:
    sequence: list[int]
    "Byte sequence"
    offset: int = 0
    "Current offset into the sequence"
    ended: bool = False
    "If the sequence has reached the end"
    losses: list[torch.Tensor] = dataclasses.field(default_factory=list)
    "List of losses for all steps"
    unweighted_losses: list[torch.Tensor] = dataclasses.field(default_factory=list)
    "List of losses for all steps before distance weighting, but after temporal weighting"

class TrainBatch:
    model_config: ModelConfig
    train_config: TrainConfig
    device: torch.device
    dtype: torch.dtype
    sequences: list[TrainSequence]
    model: RNN
    recurrent: torch.Tensor
    batch_size: int
    "Size of this batch"
    prev_output_embeddings: torch.Tensor
    "Embedding vectors (before decode) from previous step"
    prev_output_tokens: torch.Tensor
    "Concrete tokens (after decode) from previous step"
    prev_shifts: torch.Tensor
    "Shift from previous step"
    sequence_provider: SequenceProvider
    advance_rate: list[float]
    "Averages of advance rate in characters per step"

    def __init__(
        self,
        model: RNN,
        model_config: ModelConfig,
        train_config: TrainConfig,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        sequence_provider: SequenceProvider,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.dtype = dtype
        self.model = model
        self.batch_size = batch_size
        self.sequence_provider = sequence_provider

        # TODO: pre-create uncommitted mask
        # ensure state exists
        self.init_state()

    def next_sequence(self) -> list[int]:
        text = self.sequence_provider.next_sequence()
        text_bytes = text.encode('utf-8')
        short_ctx_len = self.model_config.short_ctx_len
        out_ctx_len = self.model_config.out_ctx_len
        start_pad = [ControlTokens.PAD] * (short_ctx_len - 1) + [ControlTokens.START_OF_TEXT]
        end_pad = [ControlTokens.END_OF_TEXT] + [ControlTokens.EMPTY] * (out_ctx_len - 1)
        return start_pad + list(text_bytes) + end_pad

    def init_state(self):
        "initialize internal state for new batch"
        # initialize shift to the full output length so all are marked as new
        self.prev_shifts = torch.full(
            (self.batch_size,),
            self.model_config.out_ctx_len,
            dtype=torch.int64,
            device=self.device,
        )
        self.prev_output_embeddings = torch.zeros(
            (self.batch_size, self.model_config.out_ctx_len, self.model_config.d_embed),
            dtype=self.dtype,
            device=self.device,
        )
        self.prev_output_tokens = torch.full(
            (self.batch_size, self.model_config.out_ctx_len),
            ControlTokens.EMPTY,
            dtype=torch.int64,
            device=self.device,
        )
        self.recurrent = self.model.recurrent_init.unsqueeze(0).repeat_interleave(self.batch_size, 0)

    def initialize_sequences(self):
        self.init_state()
        self.sequences = [
            TrainSequence(sequence=self.next_sequence()) for _ in range(self.batch_size)
        ]

    def detach_all(self):
        "Prepare for next batch"
        self.recurrent = self.recurrent.detach()
        for info in self.sequences:
            info.losses.clear()

    def prepare_batch(self):
        # prepare inputs and expected output
        short_ctx_len = self.model_config.short_ctx_len
        input_tokens_list: list[torch.Tensor] = []
        input_new_mask_list: list[torch.Tensor] = []
        output_list: list[torch.Tensor] = []
        for _i, info in enumerate(self.sequences):
            short_ctx_l = info.sequence[info.offset : info.offset + short_ctx_len]
            next_token_val = info.sequence[info.offset + short_ctx_len]

            if not info.ended:
                if (
                    info.prev_halted and
                    info.backspace_placeholder is None and
                    info.backspace_gap == 0 and
                    torch.bernoulli(
                        torch.tensor(self.train_config.backspace_p, device='cpu')
                    ).item() > 0
                ):
                    # set backspace
                    info.backspace_placeholder = random_token_not(len(self.tokenizer), short_ctx_l[-1])
                    info.backspace_gap = 1

                if info.backspace_placeholder is not None:
                    short_ctx_l[-1] = info.backspace_placeholder
                    next_token_val = self.tokenizer['<del>']

            short_ctx = torch.tensor(short_ctx_l, dtype=torch.int64, device='cpu', pin_memory=True)
            new_mask = torch.tensor(new_mask_l, dtype=torch.bool, device='cpu', pin_memory=True)
            next_token = torch.tensor(next_token_val, dtype=torch.int64, device='cpu', pin_memory=True)

            input_tokens_list.append(short_ctx)
            input_new_mask_list.append(new_mask)
            output_list.append(next_token)

            # print('\nprepare_batch(): sequences dump')
            # print(
            #     '  batch element:', _i,
            #     repr(''.join(self.tokenizer.IdToPiece(p) for p in short_ctx_l)),
            #     '->',
            #     repr(self.tokenizer.IdToPiece(next_token_val)),
            # )
            # print('           mask:', repr(new_mask_l))

        input_short_ctx = torch.stack(input_tokens_list, dim=0).to(self.device, non_blocking=True)
        input_new_mask = torch.stack(input_new_mask_list, dim=0).to(self.device, non_blocking=True)
        output_array = torch.stack(output_list, dim=0).to(self.device, non_blocking=True)

        return input_short_ctx, input_new_mask, output_array

    @torch.compile(disable=DISABLE_TORCH_COMPILE)
    def forward_batch(
        self,
        recurrent: torch.Tensor,
        short_ctx: torch.Tensor,
        expected_output: torch.Tensor,
    ):
        # replace some tokens in short ctx with <pad>, except for control tokens
        short_ctx = torch.where(
            (torch.full(short_ctx.shape, self.train_config.short_ctx_dropout_p).bernoulli() > 0)
                .logical_and(short_ctx < 256),
            ControlTokens.PAD,
            short_ctx,
        )

        # forward model
        next_recurrent, token_out, confidence_out = self.model(recurrent, short_ctx, new_mask)

        # calculate losses
        cross_entropy = F.cross_entropy(token_out, expected_output, reduction='none')

        step_weight = p_halt_out ** 2.5
        # if halted, override to 1
        step_weight = torch.where(did_halt, 1., step_weight)
        weighted_losses = p_not_halt * step_weight * cross_entropy

        return next_recurrent, token_out, cross_entropy, weighted_losses

    def forward_step(self, force_halt=False):
        short_ctx_len = self.model_config.short_ctx_len
        out_ctx_len = self.model_config.out_ctx_len
        input_short_ctx, input_new_mask, expected_output = self.prepare_batch()

        next_recurrent, token_out, cross_entropy, weighted_losses = \
            self.forward_batch(
                self.recurrent,
                input_short_ctx,
                expected_output,
            )

        cross_entropy_cpu = cross_entropy.detach().to('cpu', non_blocking=True)

        #print('forward_step(): p_halt', p_halt_out)
        #print('forward_step(): confidence', _confidence_out)

        if isinstance(DEBUG_RECURRENT_GRAD, list):
            next_recurrent.register_hook(lambda grad: DEBUG_RECURRENT_GRAD.append(grad))

        self.recurrent = next_recurrent
        ended_count = 0

        for i, info in enumerate(self.sequences):
            if info.ended:
                # skip finished sequences
                ended_count += 1
                continue

            info.losses.append(weighted_losses[i])

            info.prev_halted = did_halt_cpu[i]
            if info.prev_halted:
                # record unweighted loss as well
                info.halted_losses.append(cross_entropy_cpu[i])

                # check if sequence ended
                # we end when the model advances past the end padding
                if offset >= len(info.sequence) - (short_ctx_len + out_ctx_len):
                    info.ended = True
                    ended_count += 1
                else:
                    # slide shortctx to include next token
                    info.offset += 1

        return ended_count

    def iter_train_losses(self):
        for info in self.sequences:
            if len(info.losses) == 0:
                assert info.ended
                continue

            # losses for each individual ponder step should be summed
            halt_steps = max(len(info.halted_losses), 1)
            yield torch.stack(info.losses).sum() / halt_steps

    def iter_unweighted_losses(self):
        for info in self.sequences:
            if len(info.halted_losses) == 0:
                continue

            seq_mean = torch.stack(info.halted_losses).sum() / len(info.halted_losses)
            yield seq_mean

class TrainHelper:
    aim_run: aim.Run | None = None
    "Aim run instance for run tracking"
    batches: list[TrainBatch]
    "List of all batches"
    model: RNN
    model_config: ModelConfig
    train_config: TrainConfig
    device: torch.device
    dtype: torch.dtype
    sequence_provider: SequenceProvider | None

    train_loss: float = 0.0
    "Train loss for current step"

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.dtype = dtype
        self.sequence_provider = None
        self.prev_unweighted_losses = []

        self.model = RNN(model_config)
        self.model.type(self.dtype)
        self.model.to(self.device)
        self.model.train()

        self.batch_count = self.train_config.accumulate_gradients
        self.batches = []

        # 16 steps is a very arbitrary number
        self.ponder_adjust_lookback = self.train_config.batch_size * \
            self.train_config.accumulate_gradients * 16

    def prepare(self):
        assert self.sequence_provider is not None
        for _ in range(self.batch_count):
            batch = TrainBatch(
                model=self.model,
                model_config=self.model_config,
                train_config=self.train_config,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.train_config.batch_size,
                sequence_provider=self.sequence_provider,
            )
            batch.initialize_sequences()
            self.batches.append(batch)

    def track(
        self,
        value,
        name: str | None = None,
        step: int | None = None,
        epoch: int | None = None,
        *,
        context: dict | None = None
    ) -> None:
        "Track stat with aim"
        if self.aim_run is not None:
            self.aim_run.track(value, name, step, epoch, context=context) # type: ignore

    def step_all(self):
        self.train_loss = 0
        print('batch: ', end='', flush=True)
        for i, batch in enumerate(self.batches):
            self.step_single(batch)
            print(i, end=' ', flush=True)

        print('done')
        return self.train_loss

    def step_single(self, batch: TrainBatch):
        # forward step TODO:
        should_new_seq = True
        for i in range(self.train_config.truncate_steps):
            is_last_step = i == self.train_config.truncate_steps - 1
            done_count = batch.forward_step(force_halt=is_last_step)
            if done_count >= batch.batch_size / 2:
                should_new_seq = True
                break

        # run loss
        seq_proportion = batch.batch_size * self.batch_count
        train_losses = list(batch.iter_train_losses())
        train_loss = torch.stack(train_losses).sum() / seq_proportion
        self.train_loss += train_loss.item()

        # backward step
        train_loss.backward()

        if should_new_seq:
            # if enough sequences have ended, get new ones
            print('(ns)', end='')
            batch.initialize_sequences()
        else:
            # continue with current sequences
            batch.detach_all()

        global DEBUG_RECURRENT_GRAD
        if isinstance(DEBUG_RECURRENT_GRAD, list):
            import matplotlib.pyplot as plt
            DEBUG_RECURRENT_GRAD.reverse()
            x = []
            y = []
            for i, grad in enumerate(DEBUG_RECURRENT_GRAD):
                if grad is None:
                    continue
                grad = grad.to(device='cpu', dtype=torch.float32)
                x.append(i)
                # pylint: disable-next=not-callable
                y.append(torch.linalg.norm(grad, dim=-1).mean())

            print(y)
            plt.plot(x, y)
            plt.show(block=True)
            DEBUG_RECURRENT_GRAD = None
        elif DEBUG_RECURRENT_GRAD is True:
            DEBUG_RECURRENT_GRAD = []

def main():
    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')

    subcommand = sys.argv[1]
    # optimizer state to load, if any
    load_optimizer_state = None
    step = 0

    if subcommand == 'new':
        config_path = sys.argv[2]
        checkpoint_path = sys.argv[3]
        data_path = sys.argv[4]
        with open(config_path, 'rb') as f:
            init_config = json.load(f)

        model_config = ModelConfig.from_dict(init_config['model_config'])
        train_config = TrainConfig.from_dict(init_config['train_config'])

        dtype = model_config.get_dtype()
        trainer = TrainHelper(model_config, train_config, device, dtype)
        trainer.aim_run = aim.Run()
        trainer.aim_run['model_config'] = model_config.to_dict()
        trainer.aim_run['train_config'] = train_config.to_dict()

    elif subcommand == 'load':
        checkpoint_path = sys.argv[2]
        data_path = sys.argv[3]

        print('reading checkpoint')
        loaded = torch.load(checkpoint_path, map_location='cpu')
        model_config = ModelConfig.from_dict(loaded['model_config'])
        train_config = TrainConfig.from_dict(loaded['train_config'])
        trainer = TrainHelper(
            model_config, train_config, device, model_config.get_dtype()
        )
        print('loading model')
        trainer.model.load_state_dict(loaded['model_state'])

        if 'run_hash' in loaded:
            trainer.aim_run = aim.Run(run_hash=loaded['run_hash'])
        else:
            trainer.aim_run = aim.Run()
            trainer.aim_run['model_config'] = model_config.to_dict()
            trainer.aim_run['train_config'] = train_config.to_dict()

        if 'optimizer_state' in loaded:
            load_optimizer_state = loaded['optimizer_state']

        if 'last_step' in loaded:
            step = loaded['last_step']

        # no need to keep this around
        del loaded

    else:
        print('unknown subcommand:', subcommand)
        return

    # TODO: temp hack
    data_iter = filter_text(load_dataset(open(data_path, 'rb')))
    trainer.sequence_provider = SequenceProvider(data_iter)

    model = trainer.model
    optimizer = train_config.make_optimizer(
        model.named_parameters(),
        allow_fused=(device.type == 'cuda'),
    )

    if load_optimizer_state is not None:
        print('loading optimizer state')
        optimizer.load_state_dict(load_optimizer_state)
        load_optimizer_state = None

    def now_str():
        return f's{step:06.0f}-{int(datetime.datetime.now().timestamp())}'

    last_checkpoint = datetime.datetime.now()
    checkpoint_now = False
    graceful_exit = False
    def handle_signal(signum, _frame):
        nonlocal checkpoint_now, graceful_exit
        print('signal received, queueing checkpoint...')
        checkpoint_now = True

        # supposedly SIGQUIT is supposed to dump core but whatever
        if signum == signal.SIGQUIT:
            graceful_exit = True

    signal.signal(signal.SIGUSR2, handle_signal)
    signal.signal(signal.SIGQUIT, handle_signal)

    def save_checkpoint(save_to: str):
        state = {
            'model_config': model_config.to_dict(),
            'train_config': train_config.to_dict(),
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'last_step': step
        }
        if trainer.aim_run is not None:
            state['run_hash'] = trainer.aim_run.hash

        torch.save(state, save_to)

    trainer.prepare()
    while True:
        step += 1
        optimizer.zero_grad()

        # TODO: determine if truncated BPTT should be used, and if so, fix it
        # if done_count >= done_threshold or True:
        #     batch += 1

        print('\nstep:', step)
        train_loss_f, unweighted_loss_f = trainer.step_all()
        print('training loss:', train_loss_f)
        print('unweighted loss:', unweighted_loss_f)
        trainer.track(train_loss_f, name='train_loss', step=step)
        trainer.track(unweighted_loss_f, name='unweighted_loss', step=step)

        grad_norm_f = nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.clip_grad_norm,
            error_if_nonfinite=True,
        ).item()

        print('grad norm:', grad_norm_f)
        trainer.track(grad_norm_f, name='grad_norm', step=step)

        if grad_norm_f > 1e3:
            print('!!! error: norm of gradients is too high:', grad_norm_f)
            save_path = checkpoint_path + '.graderr.' + now_str()
            save_checkpoint(save_path)
            print('checkpoint saved to', save_path)
            print('the current optimizer step will be skipped')
        else:
            optimizer.step()

        if (datetime.datetime.now() - last_checkpoint).total_seconds() > 3600 or checkpoint_now:
            checkpoint_now = False
            save_path = checkpoint_path + '.' + now_str()
            save_checkpoint(save_path)
            print('checkpoint saved to', save_path)
            last_checkpoint = datetime.datetime.now()

            if graceful_exit:
                print('exiting gracefully...')
                break

if __name__ == '__main__':
    #import cProfile
    #cProfile.run('main()', sort='tottime')
    main()

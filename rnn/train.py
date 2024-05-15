import dataclasses
import datetime
import json
import signal
import sys
from typing import Any

import aim
import torch
import torch._dynamo.config
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor
from torch import nn

from .common import ModelConfig, TrainConfig
from .data import filter_text, load_dataset
from .model import RNNSequence

# TODO: types hack
SequenceProvider = Any

DISABLE_TORCH_COMPILE = False
"If torch.compile should be disabled"

# extreme hack
DEBUG_RECURRENT_GRAD: list[torch.Tensor] | None = None
def signal_grad_debug(*_args):
    global DEBUG_RECURRENT_GRAD
    DEBUG_RECURRENT_GRAD = True
signal.signal(signal.SIGUSR1, signal_grad_debug)

def confidence_loss(
    loss: torch.Tensor, confidence_logit: torch.Tensor,
    prev_mean: torch.Tensor, prev_std: torch.Tensor,
):
    SQRT_2 = torch.sqrt(torch.tensor(2.))
    # rescale loss to standard normal, negate for high loss -> low confidence
    loss_normal = -(loss - prev_mean) / prev_std
    # needs sqrt(2) due to normal/logistic regression cdf difference wrt. sigmoid/tanh
    target_confidence = F.sigmoid(loss_normal * SQRT_2)
    return F.binary_cross_entropy_with_logits(
        confidence_logit,
        target_confidence,
        reduction='none'
    )

def random_token_not(total: int, not_token: int):
    "Generate a random token id that is not the provided token"
    while True:
        token = torch.randint(0, total, tuple()).item()
        if token != not_token:
            return token

@dataclasses.dataclass
class TrainSequence:
    sequence: list[int]
    "Token sequence"
    offset: int = 0
    "Current offset into the seqeuence"
    prev_offset: int | None = None
    "Previous offset"
    ended: bool = False
    "If the sequence has reached the end"
    losses: list[torch.Tensor] = dataclasses.field(default_factory=list)
    "List of losses for all steps"
    halted_losses: list[torch.Tensor] = dataclasses.field(default_factory=list)
    "Unweighted cross entropy loss at the end of each halted step"
    confidence_losses: list[torch.Tensor] = dataclasses.field(default_factory=list)
    "History of confidence losses, for metrics"
    backspace_placeholder: int | None = None
    "If we are doing backspace, contains the bad replacement token"
    backspace_gap: int = 4
    "How many tokens to wait before doing backspace again"
    prev_halted: bool = True
    "If the previous step for this sequence halted"

class TrainBatch:
    model_config: ModelConfig
    train_config: TrainConfig
    device: torch.device
    dtype: torch.dtype
    sequences: list[TrainSequence]
    model: RNNSequence
    recurrent: torch.Tensor | None = None
    tokenizer: SentencePieceProcessor
    sequence_provider: SequenceProvider
    halted_sequences: list[int]
    "List of sequences by index which halted the last step"
    batch_size: int
    "Size of this batch"
    p_not_halt: torch.Tensor
    "Running probability that all previous ponder steps have not halted"
    # TODO: not anymore

    def __init__(
        self,
        model: RNNSequence,
        model_config: ModelConfig,
        train_config: TrainConfig,
        tokenizer: SentencePieceProcessor,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        sequence_provider: SequenceProvider,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size
        self.sequence_provider = sequence_provider

        self.pad_token = torch.tensor(tokenizer['<pad>'], dtype=torch.int64, device=self.device)
        self.shortctx_dropout_mask = torch.tensor(
            [train_config.short_ctx_dropout_p] * model_config.short_ctx_len,
            dtype=torch.float32,
            device=self.device,
        )
        self.p_not_halt = torch.ones(self.batch_size, dtype=self.dtype, device=self.device)

    def next_sequence(self) -> list[int]:
        return self.sequence_provider.next_sequence()

    def initialize_sequences(self):
        self.recurrent = self.model.make_recurrent_init() \
            .unsqueeze(0).repeat_interleave(self.batch_size, 0)
        self.sequences = [
            TrainSequence(sequence=self.next_sequence()) for _ in range(self.batch_size)
        ]
        self.p_not_halt = torch.ones(self.batch_size, dtype=self.dtype, device=self.device)
        self.halted_sequences = list(range(self.batch_size))

    def detach_all(self):
        "Prepare for next batch"
        self.recurrent = self.recurrent.detach()
        for info in self.sequences:
            info.losses.clear()
            info.halted_losses.clear()
            info.confidence_losses.clear()

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

            # calculate new_mask and update prev_offset
            if info.prev_offset is None:
                # no previous iteration
                new_mask_l = [True] * short_ctx_len
            else:
                new_mask_l = [False] * short_ctx_len
                if info.offset > info.prev_offset:
                    delta = info.offset - info.prev_offset
                    new_mask_l[-delta:] = [True] * delta
                elif info.offset < info.prev_offset:
                    delta = info.prev_offset - info.offset
                    new_mask_l[:delta] = [True] * delta
            info.prev_offset = info.offset

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
        new_mask: torch.Tensor,
        expected_output: torch.Tensor,
        min_p_halt: torch.Tensor,
        p_not_halt: torch.Tensor,
        prev_loss_mean: torch.Tensor,
        prev_loss_std: torch.Tensor,
    ):
        # replace some tokens in short ctx with <pad>, but never "new" tokens
        short_ctx = torch.where(
            (self.shortctx_dropout_mask.bernoulli() > 0)
                .unsqueeze(0)
                .logical_and(new_mask.logical_not()),
            self.pad_token,
            short_ctx,
        )

        # forward model
        next_recurrent, token_out, confidence_out = self.model(recurrent, short_ctx, new_mask)

        # calculate losses
        cross_entropy = F.cross_entropy(token_out, expected_output, reduction='none')
        confidence_losses = self.train_config.confidence_scale * \
            confidence_loss(cross_entropy, confidence_out, prev_loss_mean, prev_loss_std)
        # p_halt_out does not need grad
        p_halt_out = F.sigmoid(confidence_out.detach() + self.model_config.ponder_adjust)
        # enforce min_p_halt when determining halt only
        did_halt = torch.maximum(p_halt_out, min_p_halt).bernoulli() > 0

        step_weight = p_halt_out ** 2.5
        # if halted, override to 1
        step_weight = torch.where(did_halt, 1., step_weight)
        weighted_losses = p_not_halt * step_weight * cross_entropy + confidence_losses

        # prepare next p_not_halt
        # reset p_not_halt if halted, otherwise add current
        p_not_halt_next = torch.where(did_halt, 1., p_not_halt * (1 - step_weight))

        return next_recurrent, token_out, confidence_out, p_halt_out, \
            did_halt, cross_entropy, confidence_losses, weighted_losses, p_not_halt_next

    def forward_step(self, force_halt=False):
        short_ctx_len = self.model_config.short_ctx_len
        input_short_ctx, input_new_mask, expected_output = self.prepare_batch()

        next_recurrent, _token_out, _confidence_out, _p_halt_out, did_halt, \
            cross_entropy, confidence_losses, weighted_losses, p_not_halt_next = \
            self.forward_batch(
                self.recurrent,
                input_short_ctx,
                input_new_mask,
                expected_output,
                # prevent torch.compile from recompiling on change
                min_p_halt=torch.tensor(
                    self.train_config.min_p_halt if not force_halt else 1.,
                    device=self.device,
                    dtype=self.dtype
                ),
                p_not_halt=self.p_not_halt,
                prev_loss_mean=torch.tensor(
                    self.train_config.prev_loss_mean,
                    device=self.device,
                    dtype=self.dtype,
                ),
                prev_loss_std=torch.tensor(
                    self.train_config.prev_loss_std,
                    device=self.device,
                    dtype=self.dtype,
                ),
            )

        cross_entropy_cpu = cross_entropy.detach().to('cpu', non_blocking=True)
        confidence_losses_cpu = confidence_losses.detach().to('cpu', non_blocking=True)
        did_halt_cpu = did_halt.to('cpu', non_blocking=True)

        #print('forward_step(): p_halt', p_halt_out)
        #print('forward_step(): confidence', _confidence_out)

        if isinstance(DEBUG_RECURRENT_GRAD, list):
            next_recurrent.register_hook(lambda grad: DEBUG_RECURRENT_GRAD.append(grad))

        self.recurrent = next_recurrent
        self.p_not_halt = p_not_halt_next
        self.halted_sequences.clear()
        ended_count = 0

        for i, info in enumerate(self.sequences):
            if info.ended:
                # skip finished sequences
                ended_count += 1
                continue

            info.losses.append(weighted_losses[i])
            info.confidence_losses.append(confidence_losses_cpu[i])

            info.prev_halted = did_halt_cpu[i]
            if info.prev_halted:
                self.halted_sequences.append(i)
                # record unweighted loss as well
                info.halted_losses.append(cross_entropy_cpu[i])

                if info.backspace_placeholder is not None:
                    # shift window back by one token
                    info.offset -= 1
                    info.backspace_placeholder = None
                else:
                    # check if sequence ended
                    # we end one token before the last otherwise there is no "next" token to train on
                    if info.offset + short_ctx_len >= len(info.sequence) - 1:
                        info.ended = True
                        ended_count += 1
                    else:
                        # slide shortctx to include next token
                        info.offset += 1

                    # tick down backspace_gap
                    if info.backspace_gap > 0:
                        info.backspace_gap -= 1

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

    def iter_confidence_losses(self) -> torch.Tensor:
        # TODO: this function ought to just return a histogram or something
        sequence_losses = []
        for info in self.sequences:
            if len(info.confidence_losses) == 0:
                continue

            sequence_losses.append(
                torch.stack(info.confidence_losses).sum() / len(info.confidence_losses)
            )

        return torch.stack(sequence_losses).sum() / len(sequence_losses) \
            / self.train_config.confidence_scale

class TrainHelper:
    prev_unweighted_losses: list[float]
    "List of previous unweighted losses for adjusting ponder_loss_penalty"
    aim_run: aim.Run | None = None
    "Aim run instance for run tracking"
    batches: list[TrainBatch]
    "List of all batches"
    model: RNNSequence
    tokenizer: SentencePieceProcessor
    model_config: ModelConfig
    train_config: TrainConfig
    device: torch.device
    dtype: torch.dtype
    sequence_provider: SequenceProvider

    train_loss: float = 0.0
    "Train loss for current step"
    unweighted_loss: float = 0.0
    "Unweighted loss for current step"
    ponder_adjust_lookback: int
    "How many values to keep for calculating mean/variance for confidence"

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        tokenizer: SentencePieceProcessor,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.sequence_provider = None
        self.prev_unweighted_losses = []

        self.model = RNNSequence(model_config)
        self.model.type(self.dtype)
        self.model.to(self.device)
        self.model.train()

        self.batch_count = self.train_config.accumulate_gradients
        self.batches = []

        # 16 steps is a very arbitrary number
        self.ponder_adjust_lookback = self.train_config.batch_size * \
            self.train_config.accumulate_gradients * 16

    def prepare(self):
        for _ in range(self.batch_count):
            batch = TrainBatch(
                model=self.model,
                model_config=self.model_config,
                train_config=self.train_config,
                tokenizer=self.tokenizer,
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
        name: str = None,
        step: int = None,
        epoch: int = None,
        *,
        context: dict = None
    ) -> None:
        "Track stat with aim"
        if self.aim_run is not None:
            self.aim_run.track(value, name, step, epoch, context=context)

    def step_all(self):
        self.train_loss = 0
        self.unweighted_loss = 0
        print('batch: ', end='', flush=True)
        for i, batch in enumerate(self.batches):
            self.step_single(batch)
            print(i, end=' ', flush=True)

        print('done')
        return self.train_loss, self.unweighted_loss

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

        unweighted_losses = torch.stack(list(batch.iter_unweighted_losses()))
        self.prev_unweighted_losses += unweighted_losses.tolist()
        if len(self.prev_unweighted_losses) > self.ponder_adjust_lookback * 2:
            self.prev_unweighted_losses = self.prev_unweighted_losses[self.ponder_adjust_lookback:]
        self.unweighted_loss += (unweighted_losses.sum() / seq_proportion).item()

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

    def adjust_confidence_stats(self):
        self.train_config.prev_loss_mean = torch.mean(torch.tensor(self.prev_unweighted_losses, device='cpu')).item()
        self.train_config.prev_loss_std = torch.std(torch.tensor(self.prev_unweighted_losses, device='cpu')).item()
        print(
            'adjust_confidence_stats: mean',
            self.train_config.prev_loss_mean,
            'std',
            self.train_config.prev_loss_std,
        )

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
        tokenizer = SentencePieceProcessor()
        tokenizer.Init(model_file=model_config.tokenizer_model_path)

        dtype = model_config.get_dtype()
        trainer = TrainHelper(model_config, train_config, tokenizer, device, dtype)
        trainer.aim_run = aim.Run()
        trainer.aim_run['model_config'] = model_config.to_dict()
        trainer.aim_run['train_config'] = train_config.to_dict()

    elif subcommand == 'load':
        checkpoint_path = sys.argv[2]
        data_path = sys.argv[3]

        loaded = torch.load(checkpoint_path, map_location='cpu')
        model_config = ModelConfig.from_dict(loaded['model_config'])
        train_config = TrainConfig.from_dict(loaded['train_config'])
        tokenizer = SentencePieceProcessor()
        tokenizer.Init(model_file=model_config.tokenizer_model_path)
        trainer = TrainHelper(
            model_config, train_config, tokenizer, device, model_config.get_dtype()
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

    class TempSeqProvider:
        def __init__(self):
            self.pad_token = tokenizer['<pad>']
            self.text_start_token = tokenizer['<s>']

        def wrap_sequence(self, tokens: list[int]):
            pad_start = [self.pad_token] * (model_config.short_ctx_len - 1) + [self.text_start_token]
            return pad_start + tokens

        def next_sequence(self):
            document = next(data_iter)
            encoded = self.wrap_sequence(tokenizer.Encode(document))[:train_config.max_seq_len]
            return encoded

    trainer.sequence_provider = TempSeqProvider()

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
        # confidence_loss_f = trainer.sum_confidence_losses().item()
        # trainer.track(confidence_loss_f, name='confidence_loss', step=step)

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

        if step % 10 == 0 and len(trainer.prev_unweighted_losses) >= trainer.ponder_adjust_lookback:
            trainer.adjust_confidence_stats()
            trainer.track(trainer.train_config.prev_loss_mean, name='prev_loss_mean', step=step)
            trainer.track(trainer.train_config.prev_loss_std, name='prev_loss_std', step=step)

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

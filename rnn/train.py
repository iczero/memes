import dataclasses
import datetime
import json
import signal
import sys

import aim
import numpy as np
import torch
import torch._dynamo.config
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
from sentencepiece import SentencePieceProcessor
from torch import nn

from .common import ModelConfig, TrainConfig, random_token_not
from .data import SequenceProvider, load_dataset
from .model import RNNSequence


import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
#warnings.simplefilter('always')

DISABLE_TORCH_COMPILE = False
"If torch.compile should be disabled"
USE_PINNED_MEMORY = False
torch._dynamo.config.cache_size_limit = 128

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

@dataclasses.dataclass
class TrainSequence:
    sequence: list[int]
    "Token sequence"
    offset: int = 0
    "Current offset into the seqeuence"
    ended: bool = False
    "If the sequence has reached the end"
    losses: list[torch.Tensor] = dataclasses.field(default_factory=list)
    "List of losses for all steps"
    halted_losses: list[torch.Tensor] = dataclasses.field(default_factory=list)
    "Unweighted cross entropy loss at the end of each halted step"
    confidence_losses: list[torch.Tensor] = dataclasses.field(default_factory=list)
    "History of confidence losses, for metrics"
    prev_internal: torch.Tensor | None = None
    "Previous internal state, if not halted"
    was_backspace: bool = False
    "If the last token was randomized for backspace"

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
            [train_config.short_ctx_dropout_p] * (model_config.short_ctx_len - 1) + [0.],
            dtype=torch.float32,
            device=self.device,
        )
        self.p_not_halt = torch.ones(self.batch_size, dtype=self.dtype, device=self.device)

    def next_sequence(self) -> list[int]:
        if self.sequence_provider is None:
            raise RuntimeError('sequence_provider not provided')

        return self.sequence_provider.next_sequence()

    def next_batch(self):
        self.recurrent = self.model.make_recurrent_init() \
            .unsqueeze(0).repeat_interleave(self.batch_size, 0)
        self.sequences = [
            TrainSequence(sequence=self.next_sequence()) for _ in range(self.batch_size)
        ]
        self.p_not_halt = torch.ones(self.batch_size, dtype=self.dtype, device=self.device)
        self.halted_sequences = list(range(self.batch_size))

    @torch.compile(disable=DISABLE_TORCH_COMPILE, backend='openxla')
    def forward_input_batch(self, input_encode: torch.Tensor):
        # replace some tokens in short ctx with <pad>, but never the last
        input_encode = torch.where(
            self.shortctx_dropout_mask.bernoulli() > 0,
            self.pad_token,
            input_encode,
        )
        return self.model.input(input_encode)

    def prepare_internal_batch(self):
        short_ctx_len = self.model_config.short_ctx_len
        # construct batch for input layer
        input_batch: list[tuple[int, torch.Tensor]] = []
        input_sequences: list[torch.Tensor | None] = []
        output_list: list[torch.Tensor] = []
        for i, info in enumerate(self.sequences):
            # short_ctx_l = None
            if info.prev_internal is not None:
                assert i not in self.halted_sequences
                input_sequences.append(info.prev_internal)
            else:
                assert i in self.halted_sequences
                assert not info.ended
                # prepare batch for input layer
                short_ctx_l = info.sequence[info.offset : info.offset + short_ctx_len]
                if info.offset > 4 and torch.bernoulli(
                    torch.tensor(self.train_config.backspace_p, device='cpu')
                ).item() > 0:
                    short_ctx_l[-1] = random_token_not(len(self.tokenizer), short_ctx_l[-1])
                    info.offset -= 1
                    info.was_backspace = True
                short_ctx = torch.tensor(short_ctx_l, dtype=torch.int64, device='cpu', pin_memory=USE_PINNED_MEMORY)
                input_batch.append((i, short_ctx))
                # will be substitued later
                input_sequences.append(None)

            # grab next token
            if info.was_backspace:
                next_token = self.tokenizer['<del>']
            else:
                next_token = info.sequence[info.offset + short_ctx_len]
            next_token = torch.tensor(next_token, dtype=torch.int64, device='cpu', pin_memory=USE_PINNED_MEMORY)
            output_list.append(next_token)

            # print('\nprepare_internal_batch(): sequences dump')
            # if short_ctx_l is None:
            #     short_ctx_l = info.sequence[info.offset : info.offset + short_ctx_len]
            # print(
            #     '  batch element:', i,
            #     repr(''.join(self.tokenizer.IdToPiece(p) for p in short_ctx_l)),
            #     '->',
            #     repr(self.tokenizer.IdToPiece(next_token.item())),
            # )

        if len(input_batch) > 0:
            # run input batch
            input_encode = torch.stack([v[1] for v in input_batch], dim=0).to(self.device, non_blocking=True)
            # unfortunately, it does not, well, work
            #torch._dynamo.mark_dynamic(input_encode, 0)
            input_encode = self.forward_input_batch(input_encode).to('cpu', non_blocking=True)

            for item, encoded in zip(input_batch, input_encode):
                # input_encode first dimension is batch
                input_sequences[item[0]] = encoded

        input_array = torch.stack(input_sequences, dim=0).to(self.device, non_blocking=True)
        output_array = torch.stack(output_list, dim=0).to(self.device, non_blocking=True)

        #input_array.register_hook(lambda _grad: xm.mark_step())

        return input_array, output_array

    @torch.compile(disable=DISABLE_TORCH_COMPILE, backend='openxla')
    def forward_ponder_batch(
        self,
        recurrent: torch.Tensor,
        internal: torch.Tensor,
        expected_tokens: torch.Tensor,
        # this value should not change during training
        min_p_halt: float,
        p_not_halt: torch.Tensor,
        prev_loss_mean: torch.Tensor,
        prev_loss_std: torch.Tensor,
    ):
        next_recurrent, next_internal, token_out, confidence_out = \
            self.model.ponder(recurrent, internal)

        cross_entropy = F.cross_entropy(token_out, expected_tokens, reduction='none')
        confidence_losses = self.train_config.confidence_scale * \
            confidence_loss(cross_entropy, confidence_out, prev_loss_mean, prev_loss_std)
        p_halt_out = torch.max(
            F.sigmoid(confidence_out + self.model_config.ponder_adjust),
            torch.tensor(min_p_halt),
        )
        did_halt = p_halt_out.detach().bernoulli() > 0

        # P(halt | not previously halted) * ponder step loss
        weighted_losses = p_not_halt * p_halt_out.detach() * cross_entropy + confidence_losses

        # prepare next p_not_halt
        # reset p_not_halt if halted, otherwise add current
        p_not_halt_next = torch.where(did_halt, 1., p_not_halt * (1 - p_halt_out.detach()))

        return next_recurrent, next_internal, token_out, confidence_out, p_halt_out, \
            did_halt, cross_entropy, confidence_losses, weighted_losses, p_not_halt_next

    def forward_step(self):
        internal, expected_tokens = self.prepare_internal_batch()

        next_recurrent, next_internal, _token_out, _confidence_out, _p_halt_out, did_halt, \
            cross_entropy, confidence_losses, weighted_losses, p_not_halt_next = \
            self.forward_ponder_batch(
                self.recurrent,
                internal,
                expected_tokens,
                min_p_halt=self.train_config.min_p_halt,
                p_not_halt=self.p_not_halt,
                # prevent torch.compile from recompiling on change
                prev_loss_mean=torch.tensor(
                    self.train_config.prev_loss_mean,
                    device='cpu',
                    dtype=self.dtype,
                    pin_memory=USE_PINNED_MEMORY,
                ).to(self.device, non_blocking=True),
                prev_loss_std=torch.tensor(
                    self.train_config.prev_loss_std,
                    device='cpu',
                    dtype=self.dtype,
                    pin_memory=USE_PINNED_MEMORY,
                ).to(self.device, non_blocking=True),
            )

        cross_entropy_cpu = cross_entropy.detach().to('cpu', non_blocking=True)
        confidence_losses_cpu = confidence_losses.detach().to('cpu', non_blocking=True)
        did_halt_cpu = did_halt.to('cpu', non_blocking=True)
        next_internal_cpu = next_internal.to('cpu', non_blocking=True)
        weighted_losses_cpu = weighted_losses.to('cpu', non_blocking=True)

        #print('forward_step(): p_halt', p_halt_out)
        #print('forward_step(): confidence', _confidence_out)

        # memes?
        #next_recurrent.register_hook(lambda _grad: xm.mark_step())
        self.recurrent = next_recurrent
        self.p_not_halt = p_not_halt_next
        self.halted_sequences.clear()
        ended_count = 0

        for i, info in enumerate(self.sequences):
            if info.ended:
                # skip finished sequences
                ended_count += 1
                continue

            info.losses.append(weighted_losses_cpu[i])
            info.confidence_losses.append(confidence_losses_cpu[i])

            if did_halt_cpu[i]:
                info.prev_internal = None
                # record unweighted loss as well
                info.halted_losses.append(cross_entropy_cpu[i])

                # check if sequence ended
                # we end one token before the last otherwise there is no "next" token to train on
                if info.offset + self.model_config.short_ctx_len >= len(info.sequence) - 1:
                    info.ended = True
                    # introduce dummy internal sequence
                    info.prev_internal = torch.zeros((
                        self.model_config.short_ctx_len,
                        self.model_config.n_embed,
                    ), device='cpu', dtype=self.dtype)
                else:
                    # slide shortctx to include next token
                    if not info.was_backspace:
                        info.offset += 1
                    else:
                        info.was_backspace = False
                    self.halted_sequences.append(i)
            else:
                info.prev_internal = next_internal_cpu[i]

        return ended_count

    def truncate_backprop(self):
        "Detaches recurrent state and clears losses"
        self.recurrent = self.recurrent.detach()
        for info in self.sequences:
            info.losses.clear()
            info.halted_losses.clear()
            info.confidence_losses.clear()
            if info.prev_internal is not None:
                info.prev_internal = info.prev_internal.detach()

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
        batch.next_batch()

        # forward step
        print('forward: ', end='', flush=True)
        for i in range(self.train_config.truncate_steps):
            print(i, end=' ', flush=True)
            done_count = batch.forward_step()
            if done_count >= batch.batch_size:
                break
        print('done', end=' ', flush=True)

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

        xm.mark_step()

        # backward step
        train_loss.backward()

        xm.mark_step()

        # safe to reset if not TBPTT
        # batch.reset()

    def adjust_confidence_stats(self):
        self.train_config.prev_loss_mean = np.mean(self.prev_unweighted_losses)
        self.train_config.prev_loss_std = np.std(self.prev_unweighted_losses)
        print(
            'adjust_confidence_stats: mean',
            self.train_config.prev_loss_mean,
            'std',
            self.train_config.prev_loss_std,
        )

def main():
    #device = torch.device('cuda')
    #torch.set_float32_matmul_precision('high')
    device = xm.xla_device()

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

        loaded = torch.load(checkpoint_path)
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

    else:
        print('unknown subcommand:', subcommand)
        return

    data_file = open(data_path, 'rb')
    data_iter = load_dataset(data_file)

    def filter_text(data):
        for _count, set_name, text in data:
            if set_name not in (
                'BookCorpus2', 'Books3', 'Enron Emails', 'Gutenberg (PG-19)',
                'HackerNews', 'OpenWebText2', 'Ubuntu IRC', 'Wikipedia (en)'
            ):
                continue

            yield text

    trainer.sequence_provider = SequenceProvider(
        # TODO: actually preprocess data or something
        n_sequences=2048,
        text_loader=filter_text(data_iter),
        tokenizer=tokenizer,
        short_ctx_len=model_config.short_ctx_len,
        target_seq_len=train_config.truncate_steps,
    )

    model = trainer.model
    optimizer = train_config.make_optimizer(
        model.named_parameters(),
        allow_fused=(device.type == 'cuda'),
    )

    if load_optimizer_state is not None:
        print('loading optimizer state')
        optimizer.load_state_dict(load_optimizer_state)

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

        xm.mark_step()

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

import dataclasses
import datetime
import json
import signal
import sys

import aim
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


DISABLE_TORCH_COMPILE = False
"If torch.compile should be disabled"
USE_PINNED_MEMORY = False
torch._dynamo.config.cache_size_limit = 128

# extreme hack
DEBUG_RECURRENT_GRAD: list[torch.Tensor] | None = None
def signal_grad_debug(*_args):
    global DEBUG_RECURRENT_GRAD
    DEBUG_RECURRENT_GRAD = True
signal.signal(signal.SIGUSR1, signal_grad_debug)

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
    batch_size: int
    "Size of this batch"
    shortctx_dropout_mask: torch.Tensor

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

    def prepare_internal_batch(self):
        short_ctx_len = self.model_config.short_ctx_len
        # construct batch for input layer
        input_sequences: list[torch.Tensor | None] = []
        output_list: list[torch.Tensor] = []
        for _i, info in enumerate(self.sequences):
            # prepare batch
            if not info.ended:
                short_ctx_l = info.sequence[info.offset : info.offset + short_ctx_len]
                if info.offset > 4 and torch.bernoulli(
                    torch.tensor(self.train_config.backspace_p, device='cpu')
                ).item() > 0:
                    short_ctx_l[-1] = random_token_not(len(self.tokenizer), short_ctx_l[-1])
                    info.offset -= 1
                    info.was_backspace = True
                short_ctx = torch.tensor(short_ctx_l, dtype=torch.int64, device='cpu', pin_memory=USE_PINNED_MEMORY)
            else:
                # ended sequence, use padding
                short_ctx = self.pad_token.unsqueeze(0).repeat_interleave(short_ctx_len, 0)

            input_sequences.append(short_ctx)

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
            #     '  batch element:', _i,
            #     repr(''.join(self.tokenizer.IdToPiece(p) for p in short_ctx_l)),
            #     '->',
            #     repr(self.tokenizer.IdToPiece(next_token.item())),
            # )

        input_array = torch.stack(input_sequences, dim=0).to(self.device, non_blocking=True)
        output_array = torch.stack(output_list, dim=0).to(self.device, non_blocking=True)

        return input_array, output_array

    @torch.compile(disable=DISABLE_TORCH_COMPILE, backend='openxla')
    def forward_ponder_batch(
        self,
        recurrent: torch.Tensor,
        internal: torch.Tensor,
        expected_tokens: torch.Tensor,
    ):
        next_recurrent, next_internal, token_out, confidence_out = \
            self.model.ponder(recurrent, internal)

        cross_entropy = F.cross_entropy(token_out, expected_tokens, reduction='none')

        return next_recurrent, next_internal, token_out, confidence_out, cross_entropy

    def forward_step(self):
        internal, expected_tokens = self.prepare_internal_batch()

        next_recurrent, _next_internal, _token_out, _confidence_out, cross_entropy = \
            self.forward_ponder_batch(self.recurrent, internal, expected_tokens)

        cross_entropy = cross_entropy.detach().to('cpu', non_blocking=True)

        if isinstance(DEBUG_RECURRENT_GRAD, list):
            next_recurrent.register_hook(lambda grad: DEBUG_RECURRENT_GRAD.append(grad))

        self.recurrent = next_recurrent

        for i, info in enumerate(self.sequences):
            if info.ended:
                # skip finished sequences
                ended_count += 1
                continue

            info.losses.append(cross_entropy[i])

            # check if sequence ended
            # we end one token before the last otherwise there is no "next" token to train on
            if info.offset + self.model_config.short_ctx_len >= len(info.sequence) - 1:
                info.ended = True
            else:
                # slide shortctx to include next token
                if not info.was_backspace:
                    info.offset += 1
                else:
                    info.was_backspace = False

        return ended_count

    def truncate_backprop(self):
        "Detaches recurrent state and clears losses"
        self.recurrent = self.recurrent.detach()
        for info in self.sequences:
            info.losses.clear()

    def iter_train_losses(self):
        for info in self.sequences:
            if len(info.losses) == 0:
                assert info.ended
                continue

            # losses for each individual ponder step should be summed
            halt_steps = max(len(info.halted_losses), 1)
            yield torch.stack(info.losses).sum() / halt_steps

class TrainHelper:
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
        print('batch: ', end='', flush=True)
        for i, batch in enumerate(self.batches):
            self.step_single(batch)
            print(i, end=' ', flush=True)

        print('done')
        return self.train_loss

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

        # backward step
        train_loss.backward()

        # safe to reset if not TBPTT
        # batch.reset()
        # batch.truncate_backprop()

        global DEBUG_RECURRENT_GRAD
        if isinstance(DEBUG_RECURRENT_GRAD, list):
            #import matplotlib.pyplot as plt
            DEBUG_RECURRENT_GRAD.reverse()
            #x = []
            y = []
            for i, grad in enumerate(DEBUG_RECURRENT_GRAD):
                if grad is None:
                    continue
                grad = grad.to(device='cpu', dtype=torch.float32)
                #x.append(i)
                # pylint: disable-next=not-callable
                y.append(torch.linalg.norm(grad, dim=-1).mean())

            print(y)
            #plt.plot(x, y)
            #plt.show(block=True)
            DEBUG_RECURRENT_GRAD = None
        elif DEBUG_RECURRENT_GRAD is True:
            DEBUG_RECURRENT_GRAD = []

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
        trainer.track(train_loss_f, name='train_loss', step=step)

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

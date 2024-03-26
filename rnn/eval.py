import sys

import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor

from .model import ModelConfig, RNNSequence


DISABLE_TORCH_COMPILE = True

class InferenceHelper:
    config: ModelConfig
    "Model config"
    model: RNNSequence
    "Model object"
    tokenizer: SentencePieceProcessor
    "Tokenizer object"
    device: torch.device
    "Device for model"
    dtype: torch.dtype
    "Data type for model"
    sequence: list[int]
    "Current sequence"
    offset: int
    "Offset into sequence"
    recurrent: torch.Tensor
    "Recurrent state of model"

    def __init__(
        self,
        config: ModelConfig,
        model: RNNSequence,
        tokenizer: SentencePieceProcessor,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.sequence = []

        self.initialize()

    def initialize(self):
        self.offset = 0
        self.sequence = [self.tokenizer['<pad>']] * self.config.short_ctx_len
        self.recurrent = self.model.recurrent_init.clone()

    @property
    def offset_end(self):
        return self.offset + self.config.short_ctx_len

    def current_context(self) -> torch.Tensor:
        short_ctx = self.sequence[self.offset : self.offset_end]
        short_ctx = torch.tensor(short_ctx, dtype=torch.int64, device=self.device)
        return short_ctx

    def set_token(self, index: int, token: int):
        if index == len(self.sequence):
            self.sequence.append(token)
        else:
            self.sequence[index] = token

    def get_output(self):
        return self.sequence[self.config.short_ctx_len : self.offset_end]

    def step(self, short_ctx: torch.Tensor, max_ponder = 16) -> torch.Tensor:
        internal = self.model.input(short_ctx)
        halt = False
        ponder_count = 0
        while not halt:
            self.recurrent, internal, token_logits, confidence_logit = \
                self.model.ponder(self.recurrent, internal)
            p_halt = F.sigmoid(confidence_logit + self.config.ponder_adjust)
            halt = (torch.bernoulli(p_halt) > 0).item() or ponder_count >= max_ponder
            if halt:
                return token_logits

            ponder_count += 1

    def generate_tokens(self, limit = 256, max_ponder = 16, temperature = 1.0):
        count = 0
        while count < limit:
            count += 1
            short_ctx = self.current_context()
            token_logits = self.step(short_ctx, max_ponder)
            dist = (token_logits / temperature).softmax(-1)
            token = dist.multinomial(1).item()
            yield token, dist

            if token == self.tokenizer['<del>']:
                self.offset -= 1
                continue

            self.set_token(self.offset_end, token)
            self.offset += 1

    def feed(self, tokens: list[int]):
        for token in tokens:
            self.set_token(self.offset_end, token)
            self.offset += 1
            self.step(self.current_context())

    @torch.compile(disable=DISABLE_TORCH_COMPILE)
    def _forward_input(self, short_ctx: torch.Tensor):
        return self.model.input(short_ctx)

    @torch.compile(disable=DISABLE_TORCH_COMPILE)
    def _forward_ponder(self, recurrent: torch.Tensor, internal: torch.Tensor):
        return self.model.ponder(recurrent, internal)

    def noisy_step(self, short_ctx: torch.Tensor, max_ponder = 16):
        internal = self._forward_input(short_ctx)
        halt = False
        ponder_count = 0
        while not halt:
            self.recurrent, internal, token_logits, confidence_logit = \
                self._forward_ponder(self.recurrent, internal)
            p_halt = F.sigmoid(confidence_logit + self.config.ponder_adjust)
            halt = (torch.bernoulli(p_halt) > 0).item() or ponder_count >= max_ponder
            yield halt, ponder_count, confidence_logit, p_halt, token_logits
            if halt:
                return

            ponder_count += 1

    def noisy_generate(
        self,
        tokenizer: SentencePieceProcessor,
        prompt: str | None = None,
        max_ponder: int = 16,
        limit: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        in_tokens = [tokenizer['<s>']]
        if prompt is not None:
            in_tokens += tokenizer.Encode(prompt)

        self.sequence[self.offset_end:] = in_tokens
        input_end_offset = self.offset + len(in_tokens)
        self.offset += 1

        DELETE_TOKEN = self.tokenizer['<del>']
        is_input = True
        generated = 0
        while generated < limit:
            prev_is_input = is_input
            if is_input and self.offset >= input_end_offset:
                is_input = False

            short_ctx = self.current_context()
            selected_token = None
            for halt, _ponder_count, confidence_logit, p_halt, token_logits \
                in self.noisy_step(short_ctx, max_ponder):
                token_logits = token_logits.to('cpu')
                confidence_f = confidence_logit.item()
                p_halt_f = p_halt.item()

                tokens_topk = torch.topk(token_logits, top_k)
                tokens_topk_normalized = (tokens_topk.values / temperature).softmax(-1)
                tokens_sorted, _sort_indices = tokens_topk_normalized.sort(dim=-1, descending=True)
                tokens_indices = tokens_topk.indices[_sort_indices]
                tokens_dist_cumulative = torch.cumsum(tokens_sorted, dim=-1)
                # does not seem to be any good way to do this directly in torch
                top_p_threshold = (tokens_dist_cumulative < top_p).tolist()
                if False in top_p_threshold:
                    max_index = top_p_threshold.index(False) + 1
                    tokens_dist = tokens_sorted[..., :torch.tensor(max_index, device='cpu')]
                else:
                    tokens_dist = tokens_sorted

                if halt:
                    halt_s = '*'
                    if not is_input:
                        # hack: always pick <del> if it's the top token, ignoring sampling
                        if tokens_topk.indices[0].item() == DELETE_TOKEN:
                            selected_token = DELETE_TOKEN
                        else:
                            selected_token = tokens_indices[tokens_dist.multinomial(1)].item()
                else:
                    halt_s = ' '

                io_text = 'input: ' if is_input else 'output:'
                display = f'{io_text} conf={confidence_f:+0.4f} halt={p_halt_f:0.4f} {halt_s} | '
                if prev_is_input:
                    last_input_token = repr(tokenizer.IdToPiece(self.sequence[self.offset_end - 1]))
                    display += f'in: {last_input_token} | '
                top_tokens_short = tokens_indices[:5].tolist()
                if selected_token not in top_tokens_short and selected_token is not None:
                    top_tokens_short.append(selected_token)

                mapping_list = tokens_indices.tolist()
                selected_token_display = None
                tokens_display = []
                for token in top_tokens_short:
                    token_p = tokens_sorted[mapping_list.index(token)]
                    token_s = repr(tokenizer.IdToPiece(token))

                    token_display = f'{token_s}:{token_p:0.2f}'
                    if token == selected_token:
                        selected_token_display = f'[{token_display}]'
                    else:
                        tokens_display.append(token_display)

                if selected_token_display is not None:
                    display += selected_token_display + ' '
                display += ' '.join(tokens_display)
                print(display)

            assert is_input == (selected_token is None)
            if selected_token is not None:
                if selected_token == DELETE_TOKEN:
                    self.offset -= 1
                else:
                    self.set_token(self.offset_end, selected_token)
                    self.offset += 1
                    generated += 1
            else:
                self.offset += 1

def main():
    checkpoint_file = sys.argv[1]
    device = torch.device('cpu')
    loaded = torch.load(checkpoint_file, map_location=device)
    config = ModelConfig.from_dict(loaded['model_config'])
    model = RNNSequence(config)
    model.load_state_dict(loaded['model_state'])
    dtype = config.get_dtype()
    # needed to fix rope
    model.type(dtype)
    model.to(device)
    tokenizer = SentencePieceProcessor()
    tokenizer.Init(model_file=config.tokenizer_model_path)

    with torch.inference_mode():
        infer = InferenceHelper(config, model, tokenizer, device, dtype)
        prompt = sys.argv[2] if len(sys.argv) > 2 else None
        infer.noisy_generate(tokenizer, prompt, temperature=0.7, top_p=0.9)

        print('\n============== full sequence:')
        print(tokenizer.Decode(infer.get_output()))

if __name__ == '__main__':
    main()

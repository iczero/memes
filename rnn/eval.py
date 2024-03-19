import sys

import torch
from sentencepiece import SentencePieceProcessor

from .model import ModelConfig, RNNSequence


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

    def step(self, short_ctx: torch.Tensor, max_ponder = 16) -> torch.Tensor:
        internal = self.model.input(short_ctx)
        halt = False
        ponder_count = 0
        while not halt:
            self.recurrent, internal = self.model.ponder(self.recurrent, internal)
            token_logits, p_halt = self.model.decode(self.recurrent, internal)
            halt = (torch.bernoulli(p_halt) > 0).item() or ponder_count > max_ponder
            if halt:
                return token_logits

            print('<ponder>', end='', flush=True)
            ponder_count += 1

    def generate_tokens(self, limit = 256, max_ponder = 16, temperature = 1.0):
        count = 0
        while count < limit:
            count += 1
            short_ctx = self.current_context()
            token_logits = self.step(short_ctx, max_ponder)
            dist = (token_logits / temperature).softmax(-1)
            token = dist.multinomial(1).item()
            yield token

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

def main():
    # TODO: save this to serialized model or something
    checkpoint_file = sys.argv[1]
    loaded = torch.load(checkpoint_file)
    config = ModelConfig.from_dict(loaded['model_config'])
    model = RNNSequence(config)
    model.load_state_dict(loaded['model_state'])
    dtype = config.get_dtype()
    device = torch.device('cuda')
    # needed to fix rope
    model.type(dtype)
    model.to(device)
    tokenizer = SentencePieceProcessor()
    tokenizer.Init(model_file=config.tokenizer_model_path)

    with torch.inference_mode():
        infer = InferenceHelper(config, model, tokenizer, device, dtype)
        print('input context: ', end='', flush=True)
        infer.feed([tokenizer['<s>']])
        if len(sys.argv) > 2:
            content = tokenizer.Encode(sys.argv[2])
            for token in content:
                print(tokenizer.IdToPiece(token), end='', flush=True)
                infer.feed([token])

        print('\n\ngenerated:')

        for token in infer.generate_tokens():
            print(tokenizer.IdToPiece(token), end='', flush=True)

        print('\n\n================')
        print('full sequence:')
        print(tokenizer.Decode(infer.sequence))

if __name__ == '__main__':
    main()

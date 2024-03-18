import sys

import torch
from safetensors.torch import load_model
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
    short_ctx: list[int]
    "Short context for input"
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

        self.initialize()

    def initialize(self):
        self.short_ctx = [self.tokenizer['<pad>']] * (self.config.short_ctx_len - 1)
        self.short_ctx.append(self.tokenizer['<s>'])
        self.recurrent = self.model.recurrent_init.clone()

    def generate_tokens(self, limit = 100, _max_ponder = 16):
        count = 0
        while count < limit:
            short_ctx = torch.tensor(self.short_ctx, dtype=torch.int32, device=self.device)
            internal = self.model.input(short_ctx)
            self.recurrent, internal = self.model.ponder(self.recurrent, internal)
            token_logits = self.model.decode(self.recurrent, internal)
            token = token_logits.argmax(dim=-1).item()
            self.short_ctx.append(token)
            del self.short_ctx[0]
            count += 1
            yield token
            if count > limit:
                return

def main():
    # TODO: save this to serialized model or something
    config = ModelConfig.default()
    model = RNNSequence(config)
    load_model(model, sys.argv[1])
    dtype = next(model.parameters()).dtype
    device = torch.device('cuda')
    # needed to fix rope
    model.type(dtype)
    model.to(device)
    tokenizer = SentencePieceProcessor()
    tokenizer.Init(model_file='data/tokenizer6.model')

    with torch.inference_mode():
        infer = InferenceHelper(config, model, tokenizer, device, dtype)
        for token in infer.generate_tokens():
            print(f'{token}: {tokenizer.IdToPiece(token)}')

if __name__ == '__main__':
    main()

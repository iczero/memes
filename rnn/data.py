import json
import warnings
from collections.abc import Iterable

import numpy as np
import tqdm
import zstandard
from sentencepiece import SentencePieceProcessor

TOKENIZE_BATCH_SIZE = 128

def load_dataset(in_stream):
    dctx = zstandard.ZstdDecompressor()
    count = 0
    buffer = bytearray()
    for chunk in dctx.read_to_iter(in_stream):
        buffer += chunk
        lines = buffer.split(b'\n')
        buffer[:] = lines[-1]
        for line in lines[0:-1]:
            obj = json.loads(line)
            text = obj['text']
            set_name = obj['meta']['pile_set_name']
            if len(text) > 0:
                yield count, set_name, text
            count += 1

    if len(buffer) > 0:
        warnings.warn('dataset file did not end with newline')

class SequenceProvider:
    n_tokens: int
    states: list[Iterable[list[int]]]
    text_loader: Iterable[str]
    tokenizer: SentencePieceProcessor
    target_seq_len: int
    seq_len_high_threshold: int
    seq_len_low_threshold: int
    text_fragment_max_len: int

    pad_token: int
    text_start_token: int
    text_end_token: int

    buffered_tokens: int = 0
    sequences: list[np.ndarray]
    randgen: np.random.Generator

    def __init__(
        self,
        n_tokens: int,
        text_loader: Iterable[str],
        tokenizer: SentencePieceProcessor,
        short_ctx_len: int,
        target_seq_len: int,
        randseed: np.random.SeedSequence | None = None,
    ):
        self.text_loader = text_loader
        self.tokenizer = tokenizer
        self.short_ctx_len = short_ctx_len
        self.target_seq_len = target_seq_len
        self.n_tokens = n_tokens

        self.pad_token = self.tokenizer['<pad>']
        self.text_start_token = self.tokenizer['<s>']
        self.text_end_token = self.tokenizer['</s>']

        self.sequences = []
        self.randgen = np.random.default_rng(randseed)

    def next_document(self) -> str:
        try:
            return next(self.text_loader)
        except StopIteration as exc:
            # TODO: deal with this somehow
            raise RuntimeError('loader exhausted') from exc

    def wrap_sequence(self, tokens: list[int]):
        pad_start = [self.pad_token] * (self.short_ctx_len - 1) + [self.text_start_token]
        return pad_start + tokens

    # TODO: maybe return numpy or torch array instead?
    def next_sequence(self):
        if len(self.sequences) == 0:
            self.refresh()

        return self.wrap_sequence(self.sequences.pop().tolist())

    def refresh(self):
        self.buffered_tokens = 0
        self.sequences.clear()
        sequences = self.sequences
        pbar = tqdm.tqdm(
            leave=False, disable=None, unit='token', total=self.n_tokens,
            desc='reading sequences'
        )

        doc_count = 0
        try:
            # load sequences
            while self.buffered_tokens < self.n_tokens:
                batch = []
                for _ in range(TOKENIZE_BATCH_SIZE):
                    batch.append(self.next_document())
                tokenized = self.tokenizer.Encode(batch)
                for doc in tokenized:
                    doc_count += 1
                    pos = 0
                    while pos < len(doc):
                        tokens = doc[pos : pos + self.target_seq_len]
                        if len(tokens) < self.target_seq_len:
                            tokens.append(self.text_end_token)

                        self.buffered_tokens += len(tokens)
                        sequences.append(np.array(tokens, dtype=np.uint16))
                        pbar.update(self.buffered_tokens - pbar.n)
                        pos += self.target_seq_len

            pbar.close()
            print(f'loaded {doc_count} documents, {self.buffered_tokens} tokens')
        except:
            # don't clear bar on error
            pbar.leave = True
            raise

        # shuffle
        self.randgen.shuffle(sequences)

if __name__ == '__main__':
    pass
